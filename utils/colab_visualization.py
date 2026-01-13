"""
Visualization module for Google Colab that uses matplotlib instead of pygame.
"""

import matplotlib.pyplot as plt
import numpy as np


class ColabVisualizer:
    """
    Visualizer for the sound navigation environment using matplotlib.
    Designed specifically for Google Colab compatibility.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.fig = None
        self.ax = None
        
    def update(self, env):
        """
        Update visualization with current environment state.
        
        Args:
            env: Environment object with grid, agent_pos, and sound_sources
        """
        # Create or clear the plot
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Clear the axes
        self.ax.clear()
        
        # Create a copy of the grid for visualization
        vis_grid = env.grid.copy()
        
        # Mark the agent position if exists
        if env.agent_pos is not None:
            x, y = env.agent_pos
            vis_grid[x][y] = 2
            
        # Mark sound sources if exist
        for source in env.sound_sources:
            vis_grid[source.x][source.y] = 3
        
        # Display the grid
        im = self.ax.imshow(vis_grid, cmap='viridis', interpolation='nearest')
        self.ax.set_title(f'Grid World Visualization - Step: {getattr(env, "step_count", 0)}')
        
        # Add colorbar
        cbar = self.fig.colorbar(im, ax=self.ax, label='Cell Type (0: Empty, 1: Wall, 2: Agent, 3: Sound Source)')
        
        # Show the plot
        plt.show()
        
    def close(self):
        """Close the visualization."""
        if self.fig is not None:
            plt.close(self.fig)


def visualize_single_frame(env, title="Environment State"):
    """
    Visualize a single frame of the environment without maintaining state.
    
    Args:
        env: Environment object with grid, agent_pos, and sound_sources
        title: Title for the visualization
    """
    # Create a copy of the grid for visualization
    vis_grid = env.grid.copy()
    
    # Mark the agent position if exists
    if env.agent_pos is not None:
        x, y = env.agent_pos
        vis_grid[x][y] = 2
        
    # Mark sound sources if exist
    for source in env.sound_sources:
        vis_grid[source.x][source.y] = 3
    
    plt.figure(figsize=(8, 8))
    plt.imshow(vis_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cell Type (0: Empty, 1: Wall, 2: Agent, 3: Sound Source)')
    plt.title(title)
    plt.show()


def plot_training_progress(losses, rewards):
    """
    Plot training progress showing losses and rewards over time.
    
    Args:
        losses: List of loss values during training
        rewards: List of cumulative rewards during training
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(losses)
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    
    # Plot rewards
    ax2.plot(rewards)
    ax2.set_title('Cumulative Reward Over Episodes')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Reward')
    
    plt.tight_layout()
    plt.show()