import os
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from gymnasium import spaces
import pathlib
from scipy.linalg import solve_continuous_are, solve_discrete_are # For LQR
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import random

# --- Global Constants (Existing) ---
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
G = 9.8
NUM_EVAL_DRONES = 4
RANDOM_BOUNDS = 15.0

LOG_DIR = "./hover_drone_logs/"
MODEL_DIR = "./hover_drone_models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================================================================
# 0. NEURAL NETWORK FOR DYNAMICS AND DATA COLLECTION
# ==============================================================================

class DroneDynamicsNN(nn.Module):
    """
    Neural Network to learn the drone's dynamics: s' = NN(s, a).
    Input: state (8) + action (3) = 11 dimensions
    Output: next state (8) dimensions
    """
    def __init__(self, state_dim=8, action_dim=3):
        super(DroneDynamicsNN, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, state_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x # This represents delta_state (ds = s' - s) or next_state directly

class DynamicsLearner:
    def __init__(self, state_dim=8, action_dim=3, dt=1./240.):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DroneDynamicsNN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim
        print(f"Dynamics NN initialized on device: {self.device}")

    def collect_data(self, env: "DroneMazeEnv", num_episodes=50, steps_per_episode=500):
        """
        Collects data by applying random actions in the environment.
        Data format: (current_state, action, next_state)
        """
        print(f"Collecting dynamics data for {num_episodes} episodes...")
        data = []
        for i in range(num_episodes):
            initial_pos = np.random.uniform(-RANDOM_BOUNDS/2, RANDOM_BOUNDS/2, size=(1, 2))
            initial_pos = np.hstack([initial_pos, np.random.uniform(1.0, 5.0, size=(1, 1))])
            
            goal_pos = np.random.uniform(-RANDOM_BOUNDS/2, RANDOM_BOUNDS/2, size=(1, 2))
            goal_pos = np.hstack([goal_pos, np.random.uniform(1.0, 5.0, size=(1, 1))])

            env.reset_env(start_positions=initial_pos, goals=goal_pos)
            current_state = env.states[0] # Assume single drone for data collection
            
            for _ in range(steps_per_episode):
                # Apply random actions (scaled to drone's max capabilities)
                # Note: These ranges should match those in LQRImitationDroneEnv or HybridDroneEnv
                max_omega = np.pi # max_angle from DroneMazeEnv
                max_az = G / 2 # max_az from BaseDroneEnvLQR
                
                # Random actions for omega_x, omega_y, az
                random_action = np.array([
                    np.random.uniform(-max_omega, max_omega),
                    np.random.uniform(-max_omega, max_omega),
                    np.random.uniform(-max_az, max_az)
                ])
                
                # Store data before stepping
                data.append({
                    'state': current_state.copy(),
                    'action': random_action.copy(),
                })
                
                # Step the environment with actual physical actions
                env.step(np.array([random_action]), dt=self.dt)
                next_state = env.states[0]
                
                # Update the stored data with the next_state
                data[-1]['next_state'] = next_state.copy()
                current_state = next_state
                
                # Simple boundary check to prevent divergence too far
                if np.linalg.norm(current_state[:3]) > (RANDOM_BOUNDS + 10) or current_state[2] < 0.1:
                    break
            if i % 10 == 0:
                print(f"Collected data for episode {i+1}/{num_episodes}")
        print(f"Finished data collection. Total samples: {len(data)}")
        return data

    def train_nn(self, data, epochs=100, batch_size=64):
        print(f"Training dynamics NN for {epochs} epochs...")
        states = torch.tensor([d['state'] for d in data], dtype=torch.float32).to(self.device)
        actions = torch.tensor([d['action'] for d in data], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([d['next_state'] for d in data], dtype=torch.float32).to(self.device)

        # We're predicting the *change* in state (delta_s = s_t+1 - s_t)
        # Or, we can predict next_state directly. Let's predict next_state directly for simplicity
        # If predicting delta_s, then target = next_states - states
        # For predicting next_state, target = next_states
        
        dataset = TensorDataset(states, actions, next_states)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            for batch_states, batch_actions, batch_next_states in dataloader:
                self.optimizer.zero_grad()
                
                # Predict next state directly
                predicted_next_states = self.model(batch_states, batch_actions)
                
                loss = self.criterion(predicted_next_states, batch_next_states)
                loss.backward()
                self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        print("Dynamics NN training complete.")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Dynamics NN model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval() # Set to evaluation mode
        print(f"Dynamics NN model loaded from {path}")

    def predict_next_state(self, state, action):
        self.model.eval()
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predicted_next_state = self.model(state_tensor, action_tensor)
        return predicted_next_state.cpu().numpy()
    
    def get_jacobian(self, state_vec, action_vec):
        """
        Computes the Jacobian matrices (A and B) of the learned dynamics
        around a given state-action operating point using autograd.
        
        f(s, a) = s' (where s' is next_state)
        df/ds = A
        df/da = B
        
        state_vec: (state_dim,) numpy array
        action_vec: (action_dim,) numpy array
        
        Returns: A (state_dim, state_dim), B (state_dim, action_dim)
        """
        self.model.eval()
        
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device, requires_grad=True)
        action_tensor = torch.tensor(action_vec, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # Predict the next state using the NN
        predicted_next_state = self.model(state_tensor, action_tensor)
        
        # Compute Jacobian df/ds (A matrix)
        A_rows = []
        for i in range(self.state_dim):
            # Compute gradient of i-th output with respect to state_tensor
            grad_output_s = torch.zeros_like(predicted_next_state)
            grad_output_s[i] = 1.0
            grad_s = torch.autograd.grad(predicted_next_state, state_tensor, grad_output_s, retain_graph=True, create_graph=True)[0]
            A_rows.append(grad_s.detach().cpu().numpy())
        A_matrix = np.vstack(A_rows)
        
        # Compute Jacobian df/da (B matrix)
        B_rows = []
        for i in range(self.state_dim):
            # Compute gradient of i-th output with respect to action_tensor
            grad_output_a = torch.zeros_like(predicted_next_state)
            grad_output_a[i] = 1.0
            grad_a = torch.autograd.grad(predicted_next_state, action_tensor, grad_output_a, retain_graph=True if i < self.state_dim - 1 else False, create_graph=True)[0]
            B_rows.append(grad_a.detach().cpu().numpy())
        B_matrix = np.vstack(B_rows)
        
        return A_matrix, B_matrix


# ==============================================================================
# 1. CORE SIMULATION ENVIRONMENT (EXISTING - NO CHANGES NEEDED HERE)
# ==============================================================================
class DroneMazeEnv:
    def __init__(self, num_drones=1, use_gui=False):
        if p.isConnected(): p.disconnect()
        self.num_drones = num_drones
        self.use_gui = use_gui
        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        self.states = np.zeros((self.num_drones, 8)) # [x, y, z, vx, vy, vz, theta_x, theta_y]
        self.goals = np.zeros((self.num_drones, 3))
        self.drone_ids = []
        self.current_step = 0

    def reset_env(self, start_positions=None, goals=None):
        p.resetSimulation()
        p.setGravity(0, 0, -G)
        self.current_step = 0
        if start_positions is None:
            start_positions = np.hstack([
                np.random.uniform(-RANDOM_BOUNDS/2, RANDOM_BOUNDS/2, size=(self.num_drones, 2)), 
                np.random.uniform(1.0, 5.0, size=(self.num_drones, 1))
            ])
        if goals is None:
            goals = np.hstack([
                np.random.uniform(-RANDOM_BOUNDS/2, RANDOM_BOUNDS/2, size=(self.num_drones, 2)), 
                np.random.uniform(1.0, 5.0, size=(self.num_drones, 1))
            ])
        self.goals = np.array(goals)
        self.states = np.zeros((self.num_drones, 8))
        for i, pos in enumerate(start_positions):
            self.states[i, :3] = pos
        self._draw_goals()
        self._init_drones()
        return self.states.copy() # Return initial states

    def _draw_goals(self):
        goal_colors = [[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [1, 1, 0, 0.5], [1, 0, 1, 0.5]]
        for i, goal in enumerate(self.goals):
            color = goal_colors[i % len(goal_colors)]
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=color)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_id, basePosition=goal)

    def _init_drones(self):
        self.drone_ids = []
        script_path = pathlib.Path(__file__).parent.resolve()
        # Assume assets/iris-drone-urdf--main/iris_description/urdf/iris_pybullet.urdf is present
        urdf_path = script_path / "assets" / "iris-drone-urdf--main" / "iris_description" / "urdf" / "iris_pybullet.urdf"
        if not urdf_path.exists(): 
             # Fallback if running from a different directory or asset path is wrong
            print(f"Warning: Drone URDF not found at {urdf_path}. Trying fallback path...")
            # Example fallback: assume assets folder is in the same directory as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            fallback_urdf_path = os.path.join(script_dir, "assets", "iris-drone-urdf--main", "iris_description", "urdf", "iris_pybullet.urdf")
            if not os.path.exists(fallback_urdf_path):
                # Another common fallback: if in parent directory
                parent_dir = os.path.dirname(script_dir)
                fallback_urdf_path = os.path.join(parent_dir, "assets", "iris-drone-urdf--main", "iris_description", "urdf", "iris_pybullet.urdf")
                if not os.path.exists(fallback_urdf_path):
                     raise FileNotFoundError(f"Could not find drone URDF at {urdf_path} or common fallbacks. Please ensure 'assets/iris-drone-urdf--main/...' exists.")
            urdf_path = pathlib.Path(fallback_urdf_path)

        for i in range(self.num_drones):
            drone_id = p.loadURDF(str(urdf_path), basePosition=self.states[i, :3].tolist())
            self.drone_ids.append(drone_id)

    def step(self, actions: np.ndarray, dt=1./240.):
        """Steps the simulation with a batch of actions for all drones.
           Actions are [omega_x, omega_y, az]
        """
        new_states = []
        max_angle = np.pi / 6  # ~30 degrees, a conservative and stable physical limit

        for i, drone_id in enumerate(self.drone_ids):
            x, y, z, vx, vy, vz, theta_x, theta_y = self.states[i]
            omega_x, omega_y, az = actions[i] # actions are directly physical inputs
            
            # Update angles
            theta_x += dt * omega_x
            theta_y += dt * omega_y
            
            theta_x = np.clip(theta_x, -max_angle, max_angle)
            theta_y = np.clip(theta_y, -max_angle, max_angle)
            
            # Calculate accelerations based on angles
            acc_x = G * np.tan(theta_y)
            acc_y = G * np.tan(theta_x)
            
            # Update velocities
            vx_new = vx + dt * acc_x
            vy_new = vy + dt * acc_y
            vz_new = vz + dt * az # Direct vertical acceleration from action
            
            # Update positions
            x_new = x + dt * vx_new
            y_new = y + dt * vy_new
            z_new = z + dt * vz_new
            
            p.resetBasePositionAndOrientation(drone_id, [x_new, y_new, z_new], p.getQuaternionFromEuler([theta_x, theta_y, 0]))
            new_states.append([x_new, y_new, z_new, vx_new, vy_new, vz_new, theta_x, theta_y])
        
        self.states = np.array(new_states)
        self.current_step += 1
        return self.states

    def get_obs_for_drone(self, drone_index: int):
        drone_state = self.states[drone_index]
        goal_relative_pos = self.goals[drone_index] - drone_state[:3]
        return np.concatenate([drone_state, goal_relative_pos]).astype(np.float32)
    
    def close(self):
        if p.isConnected(): p.disconnect()


# ==============================================================================
# 2. GYM WRAPPERS (BASE LQR, IMITATION ENV, HYBRID ENV) - MODIFIED FOR NN
# ==============================================================================
class BaseDroneEnvLQR: # Not directly a gym.Env, just a base class for LQR logic
    """Holds the shared, robust LQR controller logic."""
    def __init__(self, dynamics_learner: DynamicsLearner = None):
        self.max_omega = np.pi
        self.max_az = G / 2
        self.state_dim = 8
        self.action_dim = 3
        # --- NEW: Integrate DynamicsLearner ---
        self.dynamics_learner = dynamics_learner
        if self.dynamics_learner:
            # If using NN, initial A, B can be dummy or from analytical model
            print("LQR will use NN-linearized dynamics.")
            self.K = self._calculate_lqr_gains_nn(np.zeros((8,8)), np.zeros((8,3))) # K will be updated dynamically
        else:
            # Original analytical LQR
            print("LQR will use analytical dynamics.")
            self.K = self._calculate_lqr_gains_analytical()
        # --- END NEW ---

        self.cruise_speed_limit = 2.5
        self.braking_distance = 2.0
        self.max_error_norm_for_lqr = 5.0
        self.state_dim = 8
        self.action_dim = 3

    def _calculate_lqr_gains_analytical(self):
        # Analytical A and B matrices (assuming theta_x, theta_y are small for linearization)
        # State: [x, y, z, vx, vy, vz, theta_x, theta_y] (8 states)
        # Action: [omega_x, omega_y, az] (3 actions)

        # dx = vx, dy = vy, dz = vz
        # dvx = G * theta_y (approx. tan(theta_y))
        # dvy = G * theta_x (approx. tan(theta_x))
        # dvz = az (from action)
        # dtheta_x = omega_x (from action)
        # dtheta_y = omega_y (from action)

        A = np.zeros((self.state_dim, self.state_dim))
        A[0, 3] = 1.0 # x relates to vx
        A[1, 4] = 1.0 # y relates to vy
        A[2, 5] = 1.0 # z relates to vz

        A[3, 7] = G   # vx relates to theta_y
        A[4, 6] = G   # vy relates to theta_x

        B = np.zeros((self.state_dim, self.action_dim))
        B[5, 2] = 1.0 # vz relates to az
        B[6, 0] = 1.0 # theta_x relates to omega_x
        B[7, 1] = 1.0 # theta_y relates to omega_y

        # Q and R matrices
        Q = np.diag([10, 10, 20, 3, 3, 4, 10, 10]) # State costs
        R = np.diag([0.5, 0.5, 0.1])              # Action costs (omega_x, omega_y, az)

        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        print("LQR Gain Matrix K (Analytical):\n", K)
        return K
    
    def _calculate_lqr_gains_nn(self, A_nn_discrete, B_nn_discrete): # Rename parameters for clarity
        """
        Calculates LQR gains using NN-linearized discrete A and B matrices.
        Q and R are fixed, but A and B change dynamically.
        """
        # Ensure A and B are correctly shaped
        if A_nn_discrete.shape != (self.state_dim, self.state_dim) or B_nn_discrete.shape != (self.state_dim, self.action_dim):
            print("Warning: A_nn_discrete or B_nn_discrete has incorrect shape for LQR calculation. Using dummy matrices.")
            A_nn_discrete = np.eye(self.state_dim) # Use Identity for A_d for stability in fallback
            B_nn_discrete = np.zeros((self.state_dim, self.action_dim))

        Q = np.diag([10, 10, 20, 3, 3, 4, 10, 10])
        R = np.diag([0.5, 0.5, 0.1])
        
        try:
            # --- CHANGE HERE TO solve_discrete_are ---
            P = solve_discrete_are(A_nn_discrete, B_nn_discrete, Q, R)
            # --- Calculate K for discrete LQR ---
            K = np.linalg.inv(R + B_nn_discrete.T @ P @ B_nn_discrete) @ B_nn_discrete.T @ P @ A_nn_discrete
            # print("LQR Gain Matrix K (NN-Linearized Discrete) updated.")
            return K
        except np.linalg.LinAlgError:
            print("Warning: Failed to solve DARE with NN-linearized dynamics. Using previous K or analytical K.")
            # Fallback strategy: use previous K or revert to analytical if available
            return self.K if hasattr(self, 'K') else self._calculate_lqr_gains_analytical()

    def _compute_lqr_action(self, current_state, goal_state):
        # ... (previous code) ...
        if self.dynamics_learner:
            A_nn_discrete, B_nn_discrete = self.dynamics_learner.get_jacobian(current_state, np.zeros(self.action_dim))
            


            # --- Pass discrete Jacobians directly ---
            self.K = self._calculate_lqr_gains_nn(A_nn_discrete, B_nn_discrete)
        # ... (rest of the code) ...
        # --- END NEW ---

        final_goal_state = np.zeros(self.state_dim)
        final_goal_state[:3] = goal_state[:3] # Target position
        # Target velocity, angles for hovering are typically zero

        dynamic_goal_state = final_goal_state.copy()
        pos_error_vec = current_state[:3] - goal_state[:3] # Goal is relative to actual goal, not 0,0,0
        dist_to_goal = np.linalg.norm(pos_error_vec)

        if dist_to_goal > self.braking_distance:
            direction_to_goal = -pos_error_vec / (dist_to_goal + 1e-6) # Add small epsilon to avoid division by zero
            dynamic_goal_state[3:6] = direction_to_goal * self.cruise_speed_limit # Target non-zero velocity
        # else: dynamic_goal_state[3:6] remains zero, aiming for hover

        error = current_state - dynamic_goal_state
        error_norm = np.linalg.norm(error)
        
        if error_norm > self.max_error_norm_for_lqr:
            error = error / error_norm * self.max_error_norm_for_lqr

        physical_action = -self.K @ error
        
        # Scale physical action back to [-1, 1] for the DRL agent's action space (if applicable)
        # Here, these are the *actual* physical actions [omega_x, omega_y, az]
        # and should directly be applied by the simulation.
        # But for consistency with Gym actions, we'll normalize if used as agent output.
        
        # Clip to physical limits
        omega_x_clipped = np.clip(physical_action[0], -self.max_omega, self.max_omega)
        omega_y_clipped = np.clip(physical_action[1], -self.max_omega, self.max_omega)
        az_clipped = np.clip(physical_action[2], -self.max_az, self.max_az)
        
        clipped_physical_action = np.array([omega_x_clipped, omega_y_clipped, az_clipped])
        
        # If this LQR is meant to be a *target* for DRL (imitation),
        # then we need to normalize it to [-1, 1] as DRL agent outputs.
        # If this is for direct LQR application, we use clipped_physical_action.
        
        # For imitation or reward calculation, normalize.
        normalized_action_for_drl = np.array([
            clipped_physical_action[0] / self.max_omega,
            clipped_physical_action[1] / self.max_omega,
            clipped_physical_action[2] / self.max_az
        ])
        
        return normalized_action_for_drl # Returns normalized action for DRL comparison


class LQRImitationDroneEnv(gym.Env, BaseDroneEnvLQR):
    """Diagnostic Environment: Reward is purely for imitating the LQR controller."""
    def __init__(self, use_gui=False, dynamics_learner: DynamicsLearner = None):
        gym.Env.__init__(self) # Initialize Gym Env base class
        BaseDroneEnvLQR.__init__(self, dynamics_learner) # Initialize LQR base class with NN learner
        
        self.env = DroneMazeEnv(num_drones=1, use_gui=use_gui)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.goal_state = np.zeros(8)
        self.max_steps = 240 * 15

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.env.reset_env()
        obs = self.env.get_obs_for_drone(0)
        self.goal_state[:3] = self.env.goals[0]
        return obs, {}

    def step(self, action_drl_normalized): # Action comes from DRL, normalized
        # Convert DRL's normalized action to physical action for the simulator
        physical_action = np.array([
            action_drl_normalized[0] * self.max_omega,
            action_drl_normalized[1] * self.max_omega,
            action_drl_normalized[2] * self.max_az
        ])
        
        self.env.step(np.array([physical_action]))
        current_state = self.env.states[0]
        obs = self.env.get_obs_for_drone(0)
        
        # # Compute LQR target action (also normalized)
        # lqr_target_action_normalized = self._compute_lqr_action(current_state, self.


        # Compute LQR target action (also normalized)
        lqr_target_action_normalized = self._compute_lqr_action(current_state, self.goal_state)
        
        action_difference_mse = np.mean((action_drl_normalized - lqr_target_action_normalized)**2)
        reward = np.exp(-5.0 * action_difference_mse)
        
        done = False
        info = {}
        # Termination conditions
        if current_state[2] < 0.2: # Crash if too low
            done = True
            reward = -100.0 # Significant penalty for crashing
        elif self.env.current_step >= self.max_steps:
            done = True
        elif np.linalg.norm(current_state[:3]) > (RANDOM_BOUNDS + 5): # Diverged too far
            done = True
            reward = -100.0 # Significant penalty for diverging

        if done: 
            info['final_dist'] = np.linalg.norm(current_state[:3] - self.goal_state[:3])
            
        return obs, reward, done, False, info
    
    def close(self): self.env.close()

class HybridDroneEnv(gym.Env, BaseDroneEnvLQR):
    """Main Environment: Combines travel rewards with LQR imitation for hovering."""
    def __init__(self, use_gui=False, dynamics_learner: DynamicsLearner = None):
        gym.Env.__init__(self) # Initialize Gym Env base class
        BaseDroneEnvLQR.__init__(self, dynamics_learner) # Initialize LQR base class with NN learner

        self.env = DroneMazeEnv(num_drones=1, use_gui=use_gui)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.goal_state = np.zeros(8)
        self.max_steps = 240 * 15
        
        # Reward Hyperparameters
        self.hover_radius = 2.5
        self.lqr_imitation_radius = 1.0 # Not explicitly used, but can be for defining LQR imitation region
        self.align_reward_weight = 1.5
        self.potential_reward_weight = 15.0
        self.pos_error_penalty_weight = 4.0 # Not directly used in current hybrid reward, but good to keep
        self.vel_penalty_weight = 1.0 # Not directly used
        self.lqr_imitation_weight = 2.5
        self.control_penalty_weight = 0.01
        self.success_reward = 300.0
        self.crash_penalty = -300.0
        self.prev_pos_error_norm = None

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.env.reset_env()
        obs = self.env.get_obs_for_drone(0)
        self.goal_state[:3] = self.env.goals[0]
        self.prev_pos_error_norm = np.linalg.norm(self.env.states[0][:3] - self.goal_state[:3])
        return obs, {}

    def step(self, action_drl_normalized): # Action comes from DRL, normalized
        # Convert DRL's normalized action to physical action for the simulator
        physical_action = np.array([
            action_drl_normalized[0] * self.max_omega,
            action_drl_normalized[1] * self.max_omega,
            action_drl_normalized[2] * self.max_az
        ])

        self.env.step(np.array([physical_action]))
        current_state = self.env.states[0]
        obs = self.env.get_obs_for_drone(0)
        
        pos, vel = current_state[:3], current_state[3:6]
        vel_norm = np.linalg.norm(vel)
        pos_error_norm = np.linalg.norm(pos - self.goal_state[:3])
        reward = 0

        # --- Hierarchical Reward Calculation ---
        if pos_error_norm > self.hover_radius:
            # REGIME 1: TRAVEL
            potential_reward = self.potential_reward_weight * (self.prev_pos_error_norm - pos_error_norm)
            align_reward = 0
            if vel_norm > 1e-4 and pos_error_norm > 1e-4:
                align_reward = self.align_reward_weight * np.dot(vel/(vel_norm + 1e-6), (self.goal_state[:3] - pos)/(pos_error_norm + 1e-6))
            reward += potential_reward + align_reward
        else:
            # REGIME 2: HOVER with LQR IMITATION
            # Compute LQR target action (normalized) for reward calculation
            lqr_target_action_normalized = self._compute_lqr_action(current_state, self.goal_state)
            action_diff = np.mean((action_drl_normalized - lqr_target_action_normalized)**2)
            reward += -self.lqr_imitation_weight * action_diff
        
        reward += -self.control_penalty_weight * np.sum(np.square(action_drl_normalized)) # Penalty on normalized action
        
        # --- Termination Conditions ---
        done, info = False, {}
        if pos[2] < 0.2 or np.linalg.norm(pos) > (RANDOM_BOUNDS + 5):
            reward += self.crash_penalty; done = True
        elif pos_error_norm < 0.2 and vel_norm < 0.2:
            reward += self.success_reward; done = True
        elif self.env.current_step >= self.max_steps:
            done = True
        if done: info['final_dist'] = pos_error_norm
        
        self.prev_pos_error_norm = pos_error_norm
        return obs, reward, done, False, info

    def close(self): self.env.close()

# ==============================================================================
# 3. MULTI-DRONE WRAPPER FOR INDEPENDENT DRL AGENTS (Needs update for NN)
# ==============================================================================
class MultiDroneIndependentEnv:
    """
    Wrapper that manages multiple independent DRL agents, each controlling one drone.
    Each agent receives its own observation and provides its own action.
    """
    def __init__(self, num_drones=5, use_gui=False, env_type="hybrid", dynamics_learner: DynamicsLearner = None):
        self.num_drones = num_drones
        self.use_gui = use_gui
        self.env_type = env_type
        self.dynamics_learner = dynamics_learner # Pass the dynamics learner
        
        # Create individual environment instances for each drone
        self.drone_envs = []
        for i in range(num_drones):
            if env_type == "hybrid":
                env = HybridDroneEnv(use_gui=(use_gui and i == 0), dynamics_learner=dynamics_learner)  # Only show GUI for first drone
            elif env_type == "lqr_imitation":
                env = LQRImitationDroneEnv(use_gui=(use_gui and i == 0), dynamics_learner=dynamics_learner)
            else:
                raise ValueError(f"Unknown env_type: {env_type}")
            self.drone_envs.append(env)
        
        # Create shared simulation environment
        self.shared_env = DroneMazeEnv(num_drones=num_drones, use_gui=use_gui)
        
        # Track which drones are done
        self.dones = [False] * num_drones
        self.episode_info = {}

    def reset(self):
        """Reset all drone environments and the shared simulation."""
        self.shared_env.reset_env()
        self.dones = [False] * self.num_drones
        self.episode_info = {}
        
        observations = []
        for i, drone_env in enumerate(self.drone_envs):
            # Set each drone's goal and initial state to match the shared environment
            # Note: drone_env.env is a single-drone DroneMazeEnv. We are using shared_env.states for all
            drone_env.env.states[0] = self.shared_env.states[i]
            drone_env.env.goals[0] = self.shared_env.goals[i]
            drone_env.goal_state[:3] = self.shared_env.goals[i]
            
            # Get initial observation
            obs = self.shared_env.get_obs_for_drone(i)
            observations.append(obs)
            
            # Initialize any environment-specific tracking
            if hasattr(drone_env, 'prev_pos_error_norm'):
                drone_env.prev_pos_error_norm = np.linalg.norm(
                    self.shared_env.states[i][:3] - self.shared_env.goals[i]
                )
        
        return observations

    def step(self, actions_drl_normalized):
        """
        Step all drone environments with their respective actions.
        
        Args:
            actions_drl_normalized: List of normalized actions, one for each drone
            
        Returns:
            observations: List of observations for each drone
            rewards: List of rewards for each drone
            dones: List of done flags for each drone
            infos: List of info dicts for each drone
        """
        # Scale actions for physical simulation
        physical_actions_for_simulator = []
        for i, action_norm in enumerate(actions_drl_normalized):
            if not self.dones[i]:
                drone_env = self.drone_envs[i]
                physical_action = np.array([
                    action_norm[0] * drone_env.max_omega, 
                    action_norm[1] * drone_env.max_omega, 
                    action_norm[2] * drone_env.max_az
                ])
                physical_actions_for_simulator.append(physical_action)
            else:
                # If drone is done, append zero action
                physical_actions_for_simulator.append(np.zeros(3))
        
        # Step the shared simulation
        self.shared_env.step(np.array(physical_actions_for_simulator))
        
        # Calculate rewards and observations for each drone
        observations, rewards, dones, infos = [], [], [], []
        
        for i in range(self.num_drones):
            if not self.dones[i]:
                drone_env = self.drone_envs[i]
                current_state = self.shared_env.states[i]
                obs = self.shared_env.get_obs_for_drone(i)
                
                # Calculate reward based on environment type
                reward, done, info = self._calculate_reward_and_termination(
                    i, current_state, actions_drl_normalized[i], drone_env
                )
                
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                
                if done:
                    self.dones[i] = True
                    # print(f"Drone {i+1}: Episode finished. Info: {info}") # Comment out for less verbosity
            else:
                # Drone already finished
                observations.append(np.zeros(11, dtype=np.float32))
                rewards.append(0.0)
                dones.append(True)
                infos.append({})
        
        return observations, rewards, dones, infos

    def _calculate_reward_and_termination(self, drone_idx, current_state, action_drl_normalized, drone_env):
        """Calculate reward and check termination for a specific drone."""
        pos, vel = current_state[:3], current_state[3:6]
        vel_norm = np.linalg.norm(vel)
        pos_error_norm = np.linalg.norm(pos - drone_env.goal_state[:3])
        
        reward = 0
        done = False
        info = {}
        
        if self.env_type == "hybrid":
            # Hybrid reward calculation
            if pos_error_norm > drone_env.hover_radius:
                # TRAVEL REGIME
                potential_reward = drone_env.potential_reward_weight * (
                    drone_env.prev_pos_error_norm - pos_error_norm
                )
                align_reward = 0
                if vel_norm > 1e-4 and pos_error_norm > 1e-4:
                    align_reward = drone_env.align_reward_weight * np.dot(
                        vel/(vel_norm + 1e-6), (drone_env.goal_state[:3] - pos)/(pos_error_norm + 1e-6)
                    )
                reward += potential_reward + align_reward
            else:
                # HOVER REGIME with LQR IMITATION
                lqr_action_target_normalized = drone_env._compute_lqr_action(current_state, drone_env.goal_state)
                action_diff = np.mean((action_drl_normalized - lqr_action_target_normalized)**2)
                reward += -drone_env.lqr_imitation_weight * action_diff
            
            reward += -drone_env.control_penalty_weight * np.sum(np.square(action_drl_normalized))
            
            # Termination conditions
            if pos[2] < 0.2 or np.linalg.norm(pos) > (RANDOM_BOUNDS + 5):
                reward += drone_env.crash_penalty
                done = True
            elif pos_error_norm < 0.2 and vel_norm < 0.2:
                reward += drone_env.success_reward
                done = True
            elif self.shared_env.current_step >= drone_env.max_steps:
                done = True
            
            if done:
                info['final_dist'] = pos_error_norm
            
            drone_env.prev_pos_error_norm = pos_error_norm
            
        elif self.env_type == "lqr_imitation":
            # LQR imitation reward calculation
            lqr_action_target_normalized = drone_env._compute_lqr_action(current_state, drone_env.goal_state)
            action_difference_mse = np.mean((action_drl_normalized - lqr_action_target_normalized)**2)
            reward = np.exp(-5.0 * action_difference_mse)
            
            # Termination conditions
            if (pos[2] < 0.2 or 
                self.shared_env.current_step >= drone_env.max_steps or 
                np.linalg.norm(pos) > (RANDOM_BOUNDS + 5)):
                done = True
                info['final_dist'] = pos_error_norm
        
        return reward, done, info

    def close(self):
        """Close all environments."""
        for drone_env in self.drone_envs:
            drone_env.close()
        self.shared_env.close()

# --- New: MultiDroneWrapper for Single Policy Multi-Drone (Flattened) ---
class MultiDroneWrapper(gym.Env):
    """Gym wrapper for multi-drone environment that flattens to single agent interface."""
    def __init__(self, num_drones=3, env_type="hybrid", dynamics_learner: DynamicsLearner = None):
        super().__init__()
        self.multi_env = MultiDroneIndependentEnv(num_drones=num_drones, use_gui=False, env_type=env_type, dynamics_learner=dynamics_learner)
        self.num_drones = num_drones
        
        # Observation space: concatenated observations from all drones
        single_obs_dim = 11  # Each drone has 11-dimensional observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(num_drones * single_obs_dim,), 
            dtype=np.float32
        )
        
        # Action space: concatenated actions for all drones
        single_action_dim = 3  # Each drone has 3-dimensional action
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(num_drones * single_action_dim,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        observations = self.multi_env.reset()
        # Flatten observations
        flat_obs = np.concatenate(observations)
        return flat_obs, {}

    def step(self, action_flat):
        # Reshape action back to per-drone actions
        actions_drl_normalized = action_flat.reshape(self.num_drones, 3)
        observations, rewards, dones, infos = self.multi_env.step(actions_drl_normalized)
        
        # Flatten observations and aggregate rewards
        flat_obs = np.concatenate(observations)
        total_reward = np.sum(rewards)
        all_done = all(dones)
        
        # Aggregate info
        info = {
            'individual_rewards': rewards,
            'individual_dones': dones,
            'individual_infos': infos
        }
        
        return flat_obs, total_reward, all_done, False, info

    def close(self):
        self.multi_env.close()


# ==============================================================================
# 4. TRAINING, EVALUATION, AND DIRECT LQR TEST FUNCTIONS - ENHANCED FOR MULTI-DRONE
#    (Modified to accept DynamicsLearner)
# ==============================================================================
def train_multi_drone_hybrid(dynamics_learner: DynamicsLearner = None):
    """Train a SINGLE policy to control multiple drones simultaneously."""
    print("Training a single policy to control multiple drones simultaneously...")
    NUM_ENVS = 8
    NUM_DRONES_PER_ENV = 3  # Each environment will have 3 drones
    
    # Create environments with multiple drones each - using the wrapper that flattens observations/actions
    env_fns = [
        lambda: Monitor(MultiDroneWrapper(num_drones=NUM_DRONES_PER_ENV, env_type="hybrid", dynamics_learner=dynamics_learner))
        for _ in range(NUM_ENVS)
    ]
    
    env = SubprocVecEnv(env_fns)
    norm_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    model = SAC("MlpPolicy", norm_env, 
                policy_kwargs=dict(net_arch=[512, 256, 128]),  # Larger network for multi-drone control
                verbose=1, 
                tensorboard_log=LOG_DIR + "sac_multi_drone_hybrid/", 
                buffer_size=500_000, 
                learning_rate=5e-4)
    
    model.learn(total_timesteps=5_000_000, log_interval=10)
    model.save(MODEL_DIR + "drone_sac_multi_hybrid_model")
    norm_env.save(MODEL_DIR + "vec_normalize_multi_hybrid.pkl")
    norm_env.close()

def train_independent_multi_drone(dynamics_learner: DynamicsLearner = None):
    """Train independent agents for multi-drone control using separate models."""
    print("Training independent DRL agents for multi-drone control...")
    print("Note: This approach trains separate models for each drone position.")
    
    NUM_DRONES = 3
    models = []
    
    # Train a separate model for each drone position
    for drone_idx in range(NUM_DRONES):
        print(f"\nTraining model for drone position {drone_idx + 1}...")
        
        NUM_ENVS = 8
        env_fns = [
            lambda idx=drone_idx: Monitor(SingleDroneFromMultiWrapper(drone_idx=idx, num_total_drones=NUM_DRONES, dynamics_learner=dynamics_learner))
            for _ in range(NUM_ENVS)
        ]
        
        env = SubprocVecEnv(env_fns)
        norm_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        model = SAC("MlpPolicy", norm_env, 
                    policy_kwargs=dict(net_arch=[256, 128]), 
                    verbose=1, 
                    tensorboard_log=LOG_DIR + f"sac_independent_drone_{drone_idx}/", 
                    buffer_size=300_000, 
                    learning_rate=5e-4)
        
        model.learn(total_timesteps=2_000_000, log_interval=10)
        model.save(MODEL_DIR + f"drone_sac_independent_{drone_idx}_model")
        norm_env.save(MODEL_DIR + f"vec_normalize_independent_{drone_idx}.pkl")
        norm_env.close()
        
        models.append(model)
    
    print(f"\nTrained {NUM_DRONES} independent models successfully!")

class SingleDroneFromMultiWrapper(gym.Env):
    """
    Wrapper that extracts a single drone's experience from a multi-drone environment.
    Used for training independent agents.
    """
    def __init__(self, drone_idx=0, num_total_drones=3, env_type="hybrid", dynamics_learner: DynamicsLearner = None):
        super().__init__()
        self.drone_idx = drone_idx
        self.num_total_drones = num_total_drones
        self.multi_env = MultiDroneIndependentEnv(num_drones=num_total_drones, use_gui=False, env_type=env_type, dynamics_learner=dynamics_learner)
        
        # Single drone observation and action space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        observations = self.multi_env.reset()
        return observations[self.drone_idx], {}

    def step(self, action_drl_normalized):
        # Create actions for all drones (random for others, our action for target drone)
        actions_for_multi_env = []
        for i in range(self.num_total_drones):
            if i == self.drone_idx:
                actions_for_multi_env.append(action_drl_normalized)
            else:
                # Use random actions for other drones (or could use other trained policies)
                # Ensure random actions are within [-1, 1] as expected by multi_env.step
                actions_for_multi_env.append(np.random.uniform(-0.5, 0.5, 3)) 
        
        observations, rewards, dones, infos = self.multi_env.step(actions_for_multi_env)
        
        # Return only our drone's results
        return observations[self.drone_idx], rewards[self.drone_idx], dones[self.drone_idx], False, infos[self.drone_idx]

    def close(self):
        self.multi_env.close()


def evaluate_single_policy_multi_drone(dynamics_learner: DynamicsLearner = None):
    """Evaluate a single policy controlling multiple drones simultaneously."""
    print("Evaluating single policy controlling multiple drones...")
    
    NUM_DRONES = 3
    model_path = MODEL_DIR + "drone_sac_multi_hybrid_model.zip"
    stats_path = MODEL_DIR + "vec_normalize_multi_hybrid.pkl"
    
    try:
        model = SAC.load(model_path)
        vec_normalize_stats = VecNormalize.load(stats_path, DummyVecEnv([lambda: MultiDroneWrapper(num_drones=NUM_DRONES, dynamics_learner=dynamics_learner)]))
        print(f"Loaded trained single policy model for {NUM_DRONES} drones")
    except Exception as e:
        print(f"Single policy model not found or failed to load: {e}. Please train first with train_multi_drone_hybrid()")
        return
    
    # Create evaluation environment
    eval_env = MultiDroneIndependentEnv(num_drones=NUM_DRONES, use_gui=True, env_type="hybrid", dynamics_learner=dynamics_learner)
    
    for episode in range(5):
        print(f"\n--- Single Policy Multi-Drone Episode {episode + 1} ---")
        observations = eval_env.reset()
        
        for step in range(240 * 20):  # Max 20 seconds
            # Flatten observations for the single policy
            flat_obs = np.concatenate(observations)
            norm_obs = vec_normalize_stats.normalize_obs(flat_obs)
            
            # Get single action vector for all drones
            flat_action, _ = model.predict(norm_obs, deterministic=True)
            
            # Reshape back to per-drone actions (normalized)
            actions_drl_normalized = flat_action.reshape(NUM_DRONES, 3)
            
            # Step environment
            observations, rewards, dones, infos = eval_env.step(actions_drl_normalized)
            
            if all(eval_env.dones):
                print("All drones completed!")
                break
            
            time.sleep(1./240.)
    
    eval_env.close()

def evaluate_independent_multi_drone(dynamics_learner: DynamicsLearner = None):
    """Evaluate independent policies controlling multiple drones."""
    print("Evaluating independent policies controlling multiple drones...")
    
    NUM_DRONES = 3
    models = []
    normalize_stats = []
    
    # Load all independent models
    for drone_idx in range(NUM_DRONES):
        try:
            model_path = MODEL_DIR + f"drone_sac_independent_{drone_idx}_model.zip"
            stats_path = MODEL_DIR + f"vec_normalize_independent_{drone_idx}.pkl"
            
            model = SAC.load(model_path)
            # When loading VecNormalize, provide an environment that matches the original training env
            # For SingleDroneFromMultiWrapper, the base env is like a HybridDroneEnv, so this should be fine.
            stats = VecNormalize.load(stats_path, DummyVecEnv([lambda: HybridDroneEnv(dynamics_learner=dynamics_learner)])) 
            
            models.append(model)
            normalize_stats.append(stats)
            print(f"Loaded independent model for drone {drone_idx + 1}")
        except Exception as e:
            print(f"Model for drone {drone_idx + 1} not found or failed to load: {e}. Please train first with train_independent_multi_drone()")
            return
    
    # Create evaluation environment
    eval_env = MultiDroneIndependentEnv(num_drones=NUM_DRONES, use_gui=True, env_type="hybrid", dynamics_learner=dynamics_learner)
    
    for episode in range(5):
        print(f"\n--- Independent Policies Multi-Drone Episode {episode + 1} ---")
        observations = eval_env.reset()
        
        for step in range(240 * 20):  # Max 20 seconds
            actions_drl_normalized = []
            
            # Get action from each independent model
            for i in range(NUM_DRONES):
                if not eval_env.dones[i]:
                    obs = observations[i]
                    norm_obs = normalize_stats[i].normalize_obs(obs)
                    action, _ = models[i].predict(norm_obs, deterministic=True)
                    actions_drl_normalized.append(action)
                else:
                    actions_drl_normalized.append(np.zeros(3)) # Done drones take zero action
            
            # Step environment
            observations, rewards, dones, infos = eval_env.step(actions_drl_normalized)
            
            if all(eval_env.dones):
                print("All drones completed!")
                break
            
            time.sleep(1./240.)
    
    eval_env.close()

def evaluate_multi_drone_hybrid(dynamics_learner: DynamicsLearner = None):
    """
    Evaluate multiple independent drones using a single trained model (originally single-drone model).
    This function uses the single-drone hybrid model to control multiple drones,
    each drone using the same policy.
    """
    print("Evaluating multiple independent drones with a single trained model...")
    
    # Load the trained single-drone model
    try:
        model_path = MODEL_DIR + "drone_sac_hybrid_model.zip"
        stats_path = MODEL_DIR + "vec_normalize_hybrid.pkl"
        model = SAC.load(model_path)
        vec_normalize_stats = VecNormalize.load(stats_path, DummyVecEnv([lambda: HybridDroneEnv(dynamics_learner=dynamics_learner)]))
        print("Loaded single-drone hybrid model.")
    except Exception as e:
        print(f"Single-drone hybrid model not found or failed to load: {e}. Using random actions for demonstration.")
        model = None
        vec_normalize_stats = None
    
    # Create multi-drone environment
    multi_env = MultiDroneIndependentEnv(num_drones=NUM_EVAL_DRONES, use_gui=True, env_type="hybrid", dynamics_learner=dynamics_learner)
    
    for episode in range(5):
        print(f"\n--- Multi-Drone Evaluation Episode {episode + 1} ---")
        observations = multi_env.reset()
        
        for step in range(240 * 20):  # Max 20 seconds
            actions_drl_normalized = []
            
            # Get action for each drone independently using the single model
            for i in range(NUM_EVAL_DRONES):
                if not multi_env.dones[i]:
                    obs = observations[i]
                    
                    if model is not None and vec_normalize_stats is not None:
                        # Use trained model
                        norm_obs = vec_normalize_stats.normalize_obs(obs)
                        action, _ = model.predict(norm_obs, deterministic=True)
                    else:
                        # Use random actions if model not loaded
                        action = np.random.uniform(-1, 1, 3)
                    
                    actions_drl_normalized.append(action)
                else:
                    # Drone is done, append zero action
                    actions_drl_normalized.append(np.zeros(3))
            
            # Step the environment
            observations, rewards, dones, infos = multi_env.step(actions_drl_normalized)
            
            # Check if all drones are done
            if all(multi_env.dones):
                print("All drones have completed their tasks!")
                break
            
            time.sleep(1./240.)
    
    multi_env.close()


def train_hybrid_model(dynamics_learner: DynamicsLearner = None):
    """Train single-drone hybrid model (original function), now with optional NN dynamics."""
    print("Training the full HYBRID (Travel + LQR Imitation) agent...")
    NUM_ENVS = 16
    env_fns = [lambda: Monitor(HybridDroneEnv(use_gui=False, dynamics_learner=dynamics_learner)) for _ in range(NUM_ENVS)]
    env = SubprocVecEnv(env_fns)
    norm_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    model = SAC("MlpPolicy", norm_env, policy_kwargs=dict(net_arch=[256, 128]), verbose=1, 
                tensorboard_log=LOG_DIR + "sac_hybrid/", buffer_size=500_000, learning_rate=5e-4)
    model.learn(total_timesteps=5_000_000, log_interval=10)
    model.save(MODEL_DIR + "drone_sac_hybrid_model")
    norm_env.save(MODEL_DIR + "vec_normalize_hybrid.pkl")
    norm_env.close()

def evaluate_hybrid_model(dynamics_learner: DynamicsLearner = None):
    """Evaluate single-drone hybrid model (original function), now with optional NN dynamics."""
    print("Evaluating the HYBRID policy...")
    model_path = MODEL_DIR + "drone_sac_hybrid_model.zip"
    stats_path = MODEL_DIR + "vec_normalize_hybrid.pkl"
    try:
        model = SAC.load(model_path)
        vec_normalize_stats = VecNormalize.load(stats_path, DummyVecEnv([lambda: HybridDroneEnv(dynamics_learner=dynamics_learner)]))
    except Exception as e:
        print(f"Hybrid model not found or failed to load: {e}. Cannot evaluate.")
        return

    eval_env = HybridDroneEnv(use_gui=True, dynamics_learner=dynamics_learner)
    for episode in range(10):
        print(f"\n--- Evaluation Episode {episode + 1} ---")
        obs, _ = eval_env.reset()
        for _ in range(240 * 15):
            norm_obs = vec_normalize_stats.normalize_obs(obs)
            action_drl_normalized, _ = model.predict(norm_obs, deterministic=True)
            
            # Convert normalized action back to physical for simulation
            physical_action = np.array([
                action_drl_normalized[0] * eval_env.max_omega,
                action_drl_normalized[1] * eval_env.max_omega,
                action_drl_normalized[2] * eval_env.max_az
            ])

            # Step the inner PyBullet environment directly
            # eval_env.env.step expects a list of actions for multi-drone, but here it's 1 drone
            eval_env.env.step(np.array([physical_action])) 
            obs = eval_env.env.get_obs_for_drone(0)
            time.sleep(1./240.)
    eval_env.close()

def train_lqr_imitation(dynamics_learner: DynamicsLearner = None):
    """Train LQR imitation model (original function), now with optional NN dynamics."""
    print("Training an agent to ONLY imitate the LQR controller...")
    NUM_ENVS = 16
    env_fns = [lambda: Monitor(LQRImitationDroneEnv(use_gui=False, dynamics_learner=dynamics_learner)) for _ in range(NUM_ENVS)]
    env = SubprocVecEnv(env_fns)
    norm_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    model = SAC("MlpPolicy", norm_env, policy_kwargs=dict(net_arch=[128, 128]), verbose=1, tensorboard_log=LOG_DIR + "sac_lqr_imitation/", buffer_size=200_000, learning_rate=7e-4)
    model.learn(total_timesteps=1_500_000, log_interval=10)
    model.save(MODEL_DIR + "drone_sac_lqr_imitation_model")
    norm_env.save(MODEL_DIR + "vec_normalize_lqr_imitation.pkl")
    norm_env.close()

def evaluate_lqr_imitation(dynamics_learner: DynamicsLearner = None):
    """Evaluate LQR imitation model (original function), now with optional NN dynamics."""
    print("Evaluating the PURE LQR IMITATION policy...")
    model_path = MODEL_DIR + "drone_sac_lqr_imitation_model.zip"
    stats_path = MODEL_DIR + "vec_normalize_lqr_imitation.pkl"
    try:
        model = SAC.load(model_path)
        vec_normalize_stats = VecNormalize.load(stats_path, DummyVecEnv([lambda: LQRImitationDroneEnv(dynamics_learner=dynamics_learner)]))
    except Exception as e:
        print(f"LQR imitation model not found or failed to load: {e}. Cannot evaluate.")
        return

    eval_env = LQRImitationDroneEnv(use_gui=True, dynamics_learner=dynamics_learner)
    for episode in range(10):
        print(f"\n--- Evaluation Episode {episode + 1} ---")
        obs, _ = eval_env.reset()
        for _ in range(240 * 15):
            norm_obs = vec_normalize_stats.normalize_obs(obs)
            action_drl_normalized, _ = model.predict(norm_obs, deterministic=True)
            
            # Convert normalized action back to physical for simulation
            physical_action = np.array([
                action_drl_normalized[0] * eval_env.max_omega,
                action_drl_normalized[1] * eval_env.max_omega,
                action_drl_normalized[2] * eval_env.max_az
            ])

            # Step the inner PyBullet environment directly
            eval_env.env.step(np.array([physical_action]))
            obs = eval_env.env.get_obs_for_drone(0)
            time.sleep(1./240.)
    eval_env.close()

def test_lqr_directly(dynamics_learner: DynamicsLearner = None):
    """Test LQR controller directly on single drone, now with optional NN dynamics."""
    print("--- Testing Final LQR Controller Directly ---")
    test_env = LQRImitationDroneEnv(use_gui=True, dynamics_learner=dynamics_learner) # Use LQRImitationEnv to access its _compute_lqr_action
    for episode in range(10):
        print(f"\n--- Direct LQR Test Episode {episode + 1} ---")
        test_env.reset()
        is_done = False
        for step in range(240 * 20):
            current_state = test_env.env.states[0]
            
            # _compute_lqr_action returns normalized action, but we need physical for env.step
            action_norm_lqr = test_env._compute_lqr_action(current_state, test_env.goal_state)
            
            # Convert normalized LQR action to physical action
            omega_x_physical = action_norm_lqr[0] * test_env.max_omega
            omega_y_physical = action_norm_lqr[1] * test_env.max_omega
            az_physical = action_norm_lqr[2] * test_env.max_az
            
            scaled_physical_action = np.array([[omega_x_physical, omega_y_physical, az_physical]])
            test_env.env.step(scaled_physical_action) # Step with physical action
            
            pos_error = np.linalg.norm(current_state[:3] - test_env.goal_state[:3])
            vel_norm = np.linalg.norm(current_state[3:6])
            
            if pos_error < 0.1 and vel_norm < 0.1:
                print(f"SUCCESS: Reached stable hover in {step/240.0:.2f} seconds.")
                is_done = True; break
            if np.linalg.norm(current_state[:3]) > RANDOM_BOUNDS + 5:
                print(f"FAILURE: Drone diverged."); is_done = True; break
            if current_state[2] < 0.1: # Added crash condition
                print(f"FAILURE: Drone crashed into ground."); is_done = True; break
            time.sleep(1./240.)
        if not is_done: print(f"TIMEOUT: Episode finished.")
    test_env.close()

# ==============================================================================
# 5. ADVANCED MULTI-DRONE TRAINING WITH CURRICULUM LEARNING (Needs update for NN)
# ==============================================================================
class CurriculumMultiDroneWrapper(gym.Env):
    """
    Advanced wrapper that implements curriculum learning for multi-drone scenarios.
    Starts with single drone and gradually increases difficulty.
    """
    def __init__(self, max_drones=5, env_type="hybrid", dynamics_learner: DynamicsLearner = None):
        super().__init__()
        self.max_drones = max_drones
        self.env_type = env_type
        self.dynamics_learner = dynamics_learner # Pass the dynamics learner
        self.current_num_drones = 1  # Start with single drone
        self.episode_count = 0
        self.success_rate = 0.0
        self.recent_successes = []
        
        # Create multi-drone environment
        self.multi_env = MultiDroneIndependentEnv(
            num_drones=max_drones, 
            use_gui=False, 
            env_type=env_type,
            dynamics_learner=dynamics_learner # Pass the dynamics learner
        )
        
        # Observation and action spaces for maximum number of drones
        single_obs_dim = 11
        single_action_dim = 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(max_drones * single_obs_dim,), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(max_drones * single_action_dim,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Update curriculum based on success rate
        self._update_curriculum()
        
        # Reset only the number of drones we're currently training
        self.multi_env.num_drones = self.current_num_drones
        observations = self.multi_env.reset()
        
        # Pad observations to max size
        padded_obs = self._pad_observations(observations)
        
        self.episode_count += 1
        return padded_obs, {}

    def step(self, action_flat):
        # Extract actions for active drones only
        actions_drl_normalized = action_flat.reshape(self.max_drones, 3)[:self.current_num_drones]
        
        observations, rewards, dones, infos = self.multi_env.step(actions_drl_normalized)
        
        # Update success tracking
        # A drone is 'successful' if it reaches goal and hovers stably
        # The definition of success here needs to be clear based on the env rewards/termination
        # For simplicity, if all active drones are 'done' and didn't crash
        episode_success = all(dones) and all(
            info.get('final_dist', float('inf')) < 0.5 and (info.get('individual_dones', [False]*self.current_num_drones)[i] == True) # Check if actually reached goal and not crashed/timed out
            for i, info in enumerate(infos[:self.current_num_drones]) if 'final_dist' in info
        ) # This success condition might need refinement based on exact HybridEnv success logic
        
        self.recent_successes.append(episode_success)
        if len(self.recent_successes) > 100:
            self.recent_successes.pop(0)
        
        # Pad observations and aggregate results
        padded_obs = self._pad_observations(observations)
        total_reward = np.sum(rewards)
        all_done = all(dones) # True if all active drones are done
        
        info = {
            'individual_rewards': rewards,
            'individual_dones': dones,
            'current_num_drones': self.current_num_drones,
            'success_rate': self.success_rate,
            'episode_success': episode_success # Renamed for clarity
        }
        
        return padded_obs, total_reward, all_done, False, info

    def _pad_observations(self, observations):
        """Pad observations to maximum size."""
        padded = np.zeros(self.max_drones * 11, dtype=np.float32)
        for i, obs in enumerate(observations):
            if i < self.current_num_drones: # Only pad active drones' observations
                padded[i*11:(i+1)*11] = obs
        return padded

    def _update_curriculum(self):
        """Update the number of active drones based on performance."""
        if len(self.recent_successes) >= 50:
            self.success_rate = np.mean(self.recent_successes)
            
            # Increase difficulty if doing well
            if (self.success_rate > 0.8 and 
                self.current_num_drones < self.max_drones and
                self.episode_count % 200 == 0): # Check every 200 episodes
                self.current_num_drones += 1
                self.recent_successes = []  # Reset tracking
                print(f"Curriculum: Increased to {self.current_num_drones} drones. Success rate: {self.success_rate:.2f}")
            
            # Decrease difficulty if struggling
            elif (self.success_rate < 0.3 and 
                self.current_num_drones > 1 and
                self.episode_count % 100 == 0): # Check every 100 episodes
                self.current_num_drones -= 1
                self.recent_successes = []  # Reset tracking
                print(f"Curriculum: Decreased to {self.current_num_drones} drones. Success rate: {self.success_rate:.2f}")

    def close(self):
        self.multi_env.close()

def train_curriculum_multi_drone(dynamics_learner: DynamicsLearner = None):
    """Train with curriculum learning for multi-drone scenarios, now with optional NN dynamics."""
    print("Training with curriculum learning for multi-drone control...")
    NUM_ENVS = 8
    
    env_fns = [
        lambda: Monitor(CurriculumMultiDroneWrapper(max_drones=5, env_type="hybrid", dynamics_learner=dynamics_learner))
        for _ in range(NUM_ENVS)
    ]
    
    env = SubprocVecEnv(env_fns)
    norm_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    model = SAC("MlpPolicy", norm_env, 
                policy_kwargs=dict(net_arch=[512, 256, 128]), 
                verbose=1, 
                tensorboard_log=LOG_DIR + "sac_curriculum_multi_drone/", 
                buffer_size=1_000_000, 
                learning_rate=3e-4)
    
    model.learn(total_timesteps=10_000_000, log_interval=10)
    model.save(MODEL_DIR + "drone_sac_curriculum_multi_model")
    norm_env.save(MODEL_DIR + "vec_normalize_curriculum_multi.pkl")
    norm_env.close()

# ==============================================================================
# 6. MAIN EXECUTION BLOCK - ENHANCED WITH MULTI-DRONE OPTIONS AND NN INTEGRATION
# ==============================================================================
if __name__ == "__main__":
    
    # --- Initialize Dynamics Learner (NN) ---
    dynamics_learner = DynamicsLearner(state_dim=8, action_dim=3)
    dynamics_model_path = MODEL_DIR + "drone_dynamics_nn_model.pth"

    # --- Step 1: Data Collection and NN Training ---
    # It's crucial to train the NN first before using it for LQR
    # If you have pre-collected data or a pre-trained model, skip data collection
    
    train_dynamics_nn = True # Set to True to collect data and train NN
    if train_dynamics_nn:
        print("\n--- Phase 1: Collecting Data and Training Dynamics NN ---")
        # Use a temporary environment for data collection
        data_collection_env = DroneMazeEnv(num_drones=1, use_gui=False)
        collected_data = dynamics_learner.collect_data(data_collection_env, num_episodes=100, steps_per_episode=1000)
        data_collection_env.close()
        
        dynamics_learner.train_nn(collected_data, epochs=200)
        dynamics_learner.save_model(dynamics_model_path)
    else:
        try:
            dynamics_learner.load_model(dynamics_model_path)
            print("Loaded pre-trained dynamics NN model.")
        except FileNotFoundError:
            print("Pre-trained dynamics NN model not found. Please set `train_dynamics_nn = True` to train it.")
            # If NN is not available, the LQR will fall back to analytical dynamics if `dynamics_learner` is None
            dynamics_learner = None 

    # --- Step 2: Choose LQR mode (analytical vs. NN-linearized) ---
    # Set this to True to use the trained NN for LQR linearization
    # If False, LQR will use the original analytical dynamics.
    use_nn_for_lqr = (dynamics_learner is not None)
    active_dynamics_learner = dynamics_learner if use_nn_for_lqr else None


    print("\n--- Phase 2: Running Drone Control Scenarios ---")
    print(f"LQR will use {'NN-linearized' if use_nn_for_lqr else 'Analytical'} dynamics for control.")

    # --- Uncomment the desired option to run ---
    
    # --- OPTION 1: Original Single-Drone Training and Evaluation ---
    # train_hybrid_model(dynamics_learner=active_dynamics_learner)
    # evaluate_hybrid_model(dynamics_learner=active_dynamics_learner)
    
    # --- OPTION 2: Original LQR Imitation Training and Evaluation ---
    # train_lqr_imitation(dynamics_learner=active_dynamics_learner)
    # evaluate_lqr_imitation(dynamics_learner=active_dynamics_learner)
    
    # --- OPTION 3: Original LQR Direct Testing (single drone) ---
    test_lqr_directly(dynamics_learner=active_dynamics_learner)
    
    # --- OPTION 4: Multi-Drone Independent Agent Evaluation (uses single-drone model) ---
    # evaluate_multi_drone_hybrid(dynamics_learner=active_dynamics_learner)
    
    # --- OPTION 5: Multi-Drone LQR Direct Testing ---
    # test_lqr_multi_drone(dynamics_learner=active_dynamics_learner) # Need to re-add this function
    
    # --- OPTION 6: Advanced Curriculum Learning for Multi-Drone ---
    # train_curriculum_multi_drone(dynamics_learner=active_dynamics_learner)
    
    # --- OPTION 8: Single Policy Multi-Drone Training and Evaluation ---
    # train_multi_drone_hybrid(dynamics_learner=active_dynamics_learner)
    # evaluate_single_policy_multi_drone(dynamics_learner=active_dynamics_learner)
    
    # --- OPTION 9: Independent Policies Multi-Drone Training and Evaluation ---
    # train_independent_multi_drone(dynamics_learner=active_dynamics_learner)
    # evaluate_independent_multi_drone(dynamics_learner=active_dynamics_learner)
    
    print("\n" + "="*80)
    print("MULTI-DRONE APPROACHES EXPLAINED:")
    print("="*80)
    print("Option 4: Uses ONE pre-trained single-drone model to control MULTIPLE drones")
    print("          Each drone uses the same model independently with its own obs/action")
    print("Option 8: Trains ONE policy with CONCATENATED obs/actions for ALL drones")
    print("          Single neural network outputs actions for all drones simultaneously")
    print("Option 9: Trains SEPARATE models for each drone position")
    print("           Each drone has its own specialized neural network")
    print("\nTo enable LQR with NN-linearized dynamics, ensure `train_dynamics_nn = True` is run first.")