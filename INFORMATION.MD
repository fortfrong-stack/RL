# Sound-Based Navigation System - Project Documentation

## Overview
This project implements a sound-based navigation system using reinforcement learning. The system features an agent that navigates a grid world using audio cues to complete various tasks such as finding sound sources, locating quiet areas, and following moving sound sources.

## File Structure

### Core Components
- **`core/grid_world.py`**: Contains the `GridWorld` class that represents the 25x25 grid environment and the `Agent` class that can move within the grid. Handles grid state management, rendering, and agent movement. The GridWorld class manages the state of the environment with cell types: 0 - empty, 1 - wall, 2 - agent, 3 - sound source. The Agent class defines possible actions (up, down, left, right, stay) and provides audio-based observations.

- **`core/sound_source.py`**: Defines `SoundSource` and `Wall` classes with properties like volume, frequency, and permeability. Contains the `propagate_sound` function that implements wavefront propagation algorithm for sound distribution in the environment. The `SoundSource` class has parameters for volume (determines propagation radius) and frequency (probability of emitting sound per step). The `Wall` class has permeability properties that affect how sound passes through it.

- **`core/tasks.py`**: Implements the three main navigation tasks:
  - `FindAllSourcesTask`: Find all sound sources in the environment - rewards +10 for finding new source, -0.1 per step, +50 for finding all sources
  - `FindQuietestPlaceTask`: Locate the area with minimal sound intensity - rewards based on negative current intensity, +100 for reaching quietest spot
  - `FollowMovingSourceTask`: Track a moving sound source - rewards +5 for decreasing distance, -5 for increasing distance, +100 for catching source
  Also includes a factory function `create_task_environment` for creating task-specific environments.

### Reinforcement Learning Components
- **`rl/dqn.py`**: Contains the Deep Q-Network implementation including the neural network architecture, replay buffer, and the `DQNAgentWrapper` class that manages training and inference. The DQN agent uses a multi-layer neural network with input size based on audio observation features and output size of 5 (for the 5 possible actions). The replay buffer stores experiences for experience replay training.

- **`rl/training.py`**: Implements training and evaluation functions for the DQN agents across all three tasks, including hyperparameter configurations specific to each task type. Contains functions for training agents (`train_task`), evaluating agents (`evaluate_agent`), and training all tasks (`train_all_tasks`). Each task has different learning rates and epsilon decay values optimized for the specific challenge.

### Utilities
- **`utils/audio_processing.py`**: Handles audio signal generation and feature extraction including MFCC and spectral features that serve as the agent's observations. Contains functions for generating synthetic audio signals based on intensity and frequency content, extracting MFCC features (simplified implementation), and extracting spectral features. The `get_audio_observation_features` function combines multiple feature types into a comprehensive observation vector.

- **`utils/environment_gen.py`**: Contains functions for generating random environments for training and manual environment setup for testing. The `generate_random_environment` function creates environments with random walls (20-40% of cells), random sound sources based on task type, and random agent placement. The `manual_environment_setup` function allows users to specify exact positions for walls, sources, and agent.

- **`utils/visualization.py`**: Implements Pygame-based visualization for the environment, showing agents, sound sources, walls, and visited cells. The `PygameVisualizer` class provides real-time visualization with different colors for different elements (white for empty, gray for walls, blue for agent, red for sound sources, light blue for visited cells).

### Interface
- **`interface/console_ui.py`**: Provides a console-based interface for testing the system, allowing users to interact with the environment manually or with trained agents. The interface allows users to select tasks, choose between random or manual environments, and control the agent either manually (w/a/s/d/x keys) or with a trained agent. It also integrates with the visualization module.

### Main Entry Point
- **`main.py`**: The main entry point of the application that handles command-line arguments and orchestrates training/testing of the system. Supports both command-line arguments and an interactive console menu. The main function parses arguments to determine whether to train or test, which task to use, and other configuration options.

### Test Files
- **`test_step1.py`**: Tests basic infrastructure (GridWorld and Agent) - verifies grid creation, element placement, agent movement, and boundary checks
- **`test_step2.py`**: Tests sound physics and wall interactions - verifies sound propagation, wall permeability effects, and agent audio observations
- **`test_step3.py`**: Tests audio processing and observation systems - verifies audio signal generation, feature extraction, and integration with the environment
- **`test_step4.py`**: Tests the three navigation tasks - verifies task environments, reward systems, and termination conditions

### Configuration Files
- **`README.MD`**: Contains the original project plan and implementation roadmap with detailed stage-by-stage development plan
- **`CHECKLIST.MD`**: Contains implementation checklist

## How to Run the System

### Prerequisites
- Python 3.10+
- Required packages: numpy, matplotlib, pygame, torch (PyTorch)
- Install dependencies with: `pip install numpy matplotlib pygame torch`

### Running Different Modes

1. **Interactive Console Interface** (Recommended for testing):
   ```bash
   python main.py
   ```
   This launches the main menu where you can:
   - Test environments interactively
   - Try different tasks with manual or trained agent control
   - Set up custom environments

2. **Training Mode**:
   ```bash
   python main.py --mode train --task 1
   ```
   - `--mode train`: Specifies training mode
   - `--task [1|2|3]`: Specifies which task to train (1=Find all sources, 2=Find quietest place, 3=Follow moving source)
   - Additional options:
     - `--setup [random|manual]`: Environment setup method (default: random)
     - `--episodes N`: Number of training episodes (default: 1000)
     - `--model-path PATH`: Path to save/load model

3. **Testing Mode**:
   ```bash
   python main.py --mode test --task 2
   ```
   - `--mode test`: Specifies testing mode
   - `--task [1|2|3]`: Specifies which task to test

### Running Tests
To run the step-by-step tests:
```bash
python test_step1.py  # Basic infrastructure
python test_step2.py  # Sound physics
python test_step3.py  # Audio processing
python test_step4.py  # Navigation tasks
```

## Key Features

1. **Sound Propagation Physics**: Realistic sound propagation with attenuation and wall permeability
2. **Three Navigation Tasks**: Different challenges requiring different strategies
3. **Reinforcement Learning**: DQN-based agents trained to complete tasks
4. **Audio Observations**: Agents use audio features (MFCC, spectral) as input
5. **Visualization**: Pygame-based real-time visualization of the environment
6. **Flexible Environment Generation**: Both random and manual environment setup
7. **Modular Architecture**: Clean separation of concerns with core, RL, utility, and interface layers

## Architecture Overview

The system is organized in layers:
1. **Core Layer**: Grid world, sound physics, agent movement
2. **Task Layer**: Specific navigation tasks with reward systems
3. **RL Layer**: DQN implementation and training
4. **Utility Layer**: Audio processing, environment generation, visualization
5. **Interface Layer**: Command-line and console interfaces

The agent receives audio observations based on the sound intensity and characteristics at its current position, and must navigate the environment to complete the assigned task using reinforcement learning.

## Implementation Details

### Agent Actions
The agent can perform 5 actions:
- 0: Move up
- 1: Move down
- 2: Move left
- 3: Move right
- 4: Stay in place

### Audio Feature Vector
The observation vector contains:
- MFCC coefficients (13 features)
- Spectral features (4 features: centroid, rolloff, zero-crossing rate, bandwidth)
- Raw intensity value
- Total: 18+ features depending on implementation

### Sound Propagation Algorithm
The system uses a wavefront propagation algorithm that:
- Starts from each sound source
- Propagates sound in 4 directions (up, down, left, right)
- Applies attenuation based on distance
- Considers wall permeability when sound passes through walls
- Combines contributions from multiple sources

### Training Configuration
Each task has optimized hyperparameters:
- Task 1 (Find sources): Higher learning rate (0.001), moderate epsilon decay (0.99)
- Task 2 (Quietest place): Lower learning rate (0.0005), aggressive epsilon decay (0.95)
- Task 3 (Follow source): Medium learning rate (0.0008), moderate epsilon decay (0.98)