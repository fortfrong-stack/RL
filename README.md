# Sound-Based Navigation System

This project implements a reinforcement learning system for sound-based navigation in a grid world environment. The agent learns to navigate based on audio cues to complete various tasks.

## Project Structure

```
/workspace/
├── core/                   # Core environment and agent logic
│   ├── grid_world.py       # Grid world environment and agent classes
│   ├── sound_source.py     # Sound source and propagation logic
│   └── tasks.py            # Task-specific environments
├── rl/                     # Reinforcement learning components
│   ├── dqn.py             # DQN implementation
│   └── training.py        # Training utilities
├── utils/                  # Utility functions
│   ├── audio_processing.py # Audio feature extraction
│   ├── environment_gen.py  # Environment generation
│   └── visualization.py    # Visualization utilities
├── interface/              # User interfaces
│   └── console_ui.py      # Console interface
├── main.py                # Main entry point
└── requirements.txt       # Dependencies
```

## Features

### Tasks
1. **Find all sound sources**: Locate all sound sources in the environment
2. **Find the quietest place**: Navigate to the area with lowest sound intensity
3. **Follow moving sound source**: Track a sound source that moves randomly

### Key Components

- **Grid World**: 25x25 grid with walls, agents, and sound sources
- **Sound Propagation**: Physics-based sound propagation with wall permeability
- **DQN Agent**: Deep Q-Network for learning navigation policies
- **Audio Processing**: Feature extraction from audio signals (MFCC, spectral features)

## Refactoring Improvements

The codebase has been refactored with the following improvements:

1. **Type Hints**: Added comprehensive type annotations throughout
2. **Documentation**: Enhanced docstrings with detailed parameter descriptions
3. **Code Organization**: Improved structure and readability
4. **Error Handling**: Better exception handling and validation
5. **Performance**: Optimized algorithms where possible

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Running the System

```bash
# Train an agent for a specific task
python main.py --mode train --task 1 --episodes 1000

# Launch the interactive interface
python main.py
```

### Interactive Mode

The system provides an interactive console interface accessible via:

```bash
python main.py
```

Or directly:

```bash
python -m interface.console_ui
```

## Dependencies

- numpy
- torch (PyTorch)
- pygame
- matplotlib
- tqdm

## Modules Overview

### Core (`core/`)
- `grid_world.py`: Contains the GridWorld environment and Agent classes
- `sound_source.py`: Implements sound sources, walls, and propagation physics
- `tasks.py`: Defines the three navigation tasks with specific reward structures

### Reinforcement Learning (`rl/`)
- `dqn.py`: Deep Q-Network implementation with experience replay
- `training.py`: Training loops and evaluation utilities

### Utilities (`utils/`)
- `audio_processing.py`: Audio feature extraction and signal processing
- `environment_gen.py`: Environment generation and setup utilities
- `visualization.py`: PyGame-based visualization system

## License

This project is available under the MIT License.