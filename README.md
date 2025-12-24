# Drone DRL GNC

Neural Network-based Dynamics Learning and Deep Reinforcement Learning for Drone Guidance, Navigation, and Control using PyBullet simulation.

## Overview

This project implements an advanced drone control system that combines:

- **Neural Network Dynamics Learning**: Learns drone dynamics from simulation data
- **LQR Control**: Linear Quadratic Regulator using both analytical and NN-linearized dynamics
- **Deep Reinforcement Learning**: SAC (Soft Actor-Critic) agents for autonomous control
- **Imitation Learning**: DRL agents learn to imitate optimal LQR behavior
- **Multi-Drone Control**: Support for multiple drones with various control strategies

## Features

### 1. Neural Network Dynamics Model
- Learns the drone's state transition function: `s' = NN(s, a)`
- Computes Jacobians for LQR linearization via automatic differentiation
- State: `[x, y, z, vx, vy, vz, theta_x, theta_y]` (8 dimensions)
- Action: `[omega_x, omega_y, az]` (3 dimensions)

### 2. LQR Controller
- **Analytical Mode**: Uses linearized physics equations
- **NN-Linearized Mode**: Uses Jacobians from the learned dynamics model
- Supports both continuous-time (CARE) and discrete-time (DARE) Riccati equations

### 3. Training Environments

| Environment | Description |
|------------|-------------|
| `LQRImitationDroneEnv` | Reward based on matching LQR controller actions |
| `HybridDroneEnv` | Combines travel rewards (far from goal) with LQR imitation (near goal) |
| `MultiDroneWrapper` | Single policy controlling multiple drones |
| `CurriculumMultiDroneWrapper` | Progressive difficulty increase |

### 4. Multi-Drone Strategies
- **Shared Policy**: One neural network controls all drones simultaneously
- **Independent Policies**: Separate models for each drone position
- **Curriculum Learning**: Starts with 1 drone, scales up based on success rate

## Requirements

```bash
pip install gymnasium numpy pybullet torch scipy stable-baselines3
```

## Project Structure

```
.
├── NNDyn_ImitationLQR.py              # Main implementation
├── assets/
│   ├── iris-drone-urdf--main/         # Iris drone URDF model
│   │   └── iris_description/
│   │       ├── meshes/                # 3D mesh files (.dae)
│   │       └── urdf/
│   │           └── iris_pybullet.urdf
│   ├── quadrotor.urdf                 # Alternative quadrotor model
│   └── quadrotor2.urdf
└── hover_drone_models/                # Pre-trained models
    ├── drone_dynamics_nn_model.pth    # Neural network dynamics model
    ├── drone_sac_hybrid_model.zip     # SAC hybrid policy
    ├── drone_sac_lqr_imitation_model.zip
    └── vec_normalize_*.pkl            # Normalization statistics
```

**Note**: `hover_drone_logs/` directory is created during training for TensorBoard logs.

## Usage

### Quick Start

```python
python NNDyn_ImitationLQR.py
```

By default, this will:
1. Collect dynamics data from simulation
2. Train the neural network dynamics model
3. Test LQR control with NN-linearized dynamics

### Training Options

Edit the `__main__` block to select different training modes:

```python
# Single-drone hybrid training
train_hybrid_model(dynamics_learner=active_dynamics_learner)

# LQR imitation training
train_lqr_imitation(dynamics_learner=active_dynamics_learner)

# Multi-drone with single policy
train_multi_drone_hybrid(dynamics_learner=active_dynamics_learner)

# Curriculum learning
train_curriculum_multi_drone(dynamics_learner=active_dynamics_learner)
```

### Evaluation

```python
# Test LQR controller directly
test_lqr_directly(dynamics_learner=active_dynamics_learner)

# Evaluate trained hybrid model
evaluate_hybrid_model(dynamics_learner=active_dynamics_learner)

# Multi-drone evaluation
evaluate_multi_drone_hybrid(dynamics_learner=active_dynamics_learner)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DroneMazeEnv (PyBullet)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Drone 1   │  │   Drone 2   │  │   Drone N   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   DynamicsLearner (NN)                      │
│  • Predicts next state from current state + action          │
│  • Provides Jacobians (A, B matrices) for LQR              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    BaseDroneEnvLQR                          │
│  • Computes LQR gains K from A, B matrices                  │
│  • Generates optimal control actions                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SAC Agent (Stable-Baselines3)                  │
│  • Learns to imitate LQR in hover regime                    │
│  • Learns goal-seeking behavior in travel regime            │
└─────────────────────────────────────────────────────────────┘
```

## Reward Structure (Hybrid Environment)

**Travel Regime** (distance > 2.5m from goal):
- Potential-based reward for moving toward goal
- Velocity alignment bonus

**Hover Regime** (distance < 2.5m from goal):
- LQR imitation reward (minimize action difference from LQR)
- Control effort penalty

**Terminal Rewards**:
- Success: +300 (position error < 0.2m, velocity < 0.2m/s)
- Crash: -300 (altitude < 0.2m or out of bounds)

## Configuration

Key parameters in the code:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `G` | 9.8 | Gravity constant |
| `RANDOM_BOUNDS` | 15.0 | Environment bounds |
| `max_steps` | 3600 | Max steps per episode (15 seconds) |
| `hover_radius` | 2.5 | Distance threshold for hover mode |
| `cruise_speed_limit` | 2.5 | Target velocity during travel |

## Monitoring

Training progress can be monitored with TensorBoard:

```bash
tensorboard --logdir=./hover_drone_logs/
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{drone_drl_gnc,
  author = {Rpirayesh},
  title = {Drone DRL GNC: Neural Network Dynamics and Deep Reinforcement Learning for Drone Control},
  year = {2024},
  url = {https://github.com/Rpirayesh/Drone_DRL_GNC}
}
```
