"""
Utility functions for generating environments for training and testing.
"""

import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    from .grid_world import GridWorld, Agent
    from .sound_source import SoundSource, Wall
    from .tasks import create_task_environment
except ImportError:
    from core.grid_world import GridWorld, Agent
    from core.sound_source import SoundSource, Wall
    from core.tasks import create_task_environment


def get_random_valid_position(grid, exclude_positions=None):
    """
    Get a random valid position that is not occupied by walls or other entities.
    
    Args:
        grid: The grid world
        exclude_positions: List of positions to exclude
    
    Returns:
        Tuple (x, y) of valid position or None if no valid position found
    """
    if exclude_positions is None:
        exclude_positions = []
        
    height, width = grid.shape
    valid_positions = []
    
    for x in range(height):
        for y in range(width):
            if grid[x][y] == 0 and (x, y) not in exclude_positions:
                valid_positions.append((x, y))
                
    if not valid_positions:
        return None
        
    return random.choice(valid_positions)


def generate_random_environment(task_type, width=25, height=25):
    """
    Generate a random environment for training based on task type.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        width: Width of the grid
        height: Height of the grid
    
    Returns:
        Environment instance for the specified task
    """
    # Create environment based on task type
    if task_type == 1:  # Find all sources
        num_sources = random.randint(1, 5)
        env = create_task_environment(task_type, width=width, height=height, num_sources=num_sources)
    elif task_type == 2:  # Find quietest place
        num_sources = random.randint(1, 5)
        env = create_task_environment(task_type, width=width, height=height, num_sources=num_sources)
    elif task_type == 3:  # Follow moving source
        env = create_task_environment(task_type, width=width, height=height, move_interval=random.randint(5, 15))
    else:
        raise ValueError(f"Invalid task type: {task_type}")
    
    # Create the grid world
    grid_world = env.grid_world
    
    # Add random walls (20-40% of cells)
    total_cells = width * height
    num_walls = random.randint(int(total_cells * 0.2), int(total_cells * 0.4))
    
    walls_added = 0
    while walls_added < num_walls:
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        if grid_world.grid[x][y] == 0:  # Only add walls to empty cells
            permeability = random.uniform(0.25, 1.0)
            grid_world.place_wall(x, y, permeability)
            walls_added += 1
    
    # Place agent randomly in an empty space
    placed = False
    while not placed:
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        if grid_world.grid[x][y] == 0:
            grid_world.place_agent(x, y)
            env.agent = Agent(x, y)
            placed = True
    
    # Place sound sources based on task type
    if task_type in [1, 2]:  # Multiple sources for tasks 1 and 2
        sources_to_place = env.num_sources if hasattr(env, 'num_sources') else random.randint(1, 5)
        for _ in range(sources_to_place):
            placed = False
            while not placed:
                x, y = random.randint(0, width-1), random.randint(0, height-1)
                if grid_world.grid[x][y] == 0 and (x, y) != env.agent.get_position():
                    volume = random.uniform(0.3, 1.0)
                    frequency = random.uniform(0.2, 1.0)
                    source = SoundSource(x, y, volume, frequency)
                    grid_world.place_sound_source(source)
                    placed = True
    else:  # Single source for task 3
        placed = False
        while not placed:
            x, y = random.randint(0, width-1), random.randint(0, height-1)
            if grid_world.grid[x][y] == 0 and (x, y) != env.agent.get_position():
                volume = 0.8  # Higher volume for moving source
                frequency = 0.6
                source = SoundSource(x, y, volume, frequency)
                grid_world.place_sound_source(source)
                placed = True
    
    return env


def manual_environment_setup(task_type, width=25, height=25):
    """
    Manually set up an environment for testing purposes.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        width: Width of the grid
        height: Height of the grid
    
    Returns:
        Environment instance for the specified task
    """
    # Create environment based on task type
    if task_type == 1:  # Find all sources
        num_sources = int(input("Number of sources (1-5): "))
        env = create_task_environment(task_type, width=width, height=height, num_sources=num_sources)
    elif task_type == 2:  # Find quietest place
        num_sources = int(input("Number of sources (1-5): "))
        env = create_task_environment(task_type, width=width, height=height, num_sources=num_sources)
    elif task_type == 3:  # Follow moving source
        move_interval = int(input("Source move interval (default 10): ") or "10")
        env = create_task_environment(task_type, width=width, height=height, move_interval=move_interval)
    else:
        raise ValueError(f"Invalid task type: {task_type}")
    
    grid_world = env.grid_world
    
    # Input walls
    num_walls = int(input("Number of walls: "))
    for i in range(num_walls):
        x = int(input(f"Wall {i+1} X (0-{width-1}): "))
        y = int(input(f"Wall {i+1} Y (0-{height-1}): "))
        perm = float(input("Permeability (0.25-1.0): "))
        grid_world.place_wall(x, y, perm)
    
    # Input sources
    if task_type in [1, 2]:
        num_sources = int(input("Number of sources (1-5): "))
        for i in range(num_sources):
            x = int(input(f"Source {i+1} X (0-{width-1}): "))
            y = int(input(f"Source {i+1} Y (0-{height-1}): "))
            volume = float(input("Volume (0.1-1.0): "))
            frequency = float(input("Frequency (0.1-1.0): "))
            source = SoundSource(x, y, volume, frequency)
            grid_world.place_sound_source(source)
    else:  # Task 3 - single source
        x = int(input("Moving source X (0-{width-1}): "))
        y = int(input("Moving source Y (0-{height-1}): "))
        volume = float(input("Volume (0.1-1.0): "))
        frequency = float(input("Frequency (0.1-1.0): "))
        source = SoundSource(x, y, volume, frequency)
        grid_world.place_sound_source(source)
    
    # Input agent position
    agent_x = int(input(f"Agent X (0-{width-1}): "))
    agent_y = int(input(f"Agent Y (0-{height-1}): "))
    grid_world.place_agent(agent_x, agent_y)
    env.agent = Agent(agent_x, agent_y)
    
    return env