"""
Visualization module for the sound-based navigation environment.
Uses Pygame for interactive visualization.
"""

import pygame
import numpy as np
import sys
import os

# Add parent directory to path to import from core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    from core.grid_world import GridWorld, Agent
    from core.sound_source import SoundSource, Wall
    from utils.audio_processing import get_audio_observation_features
except ImportError:
    from core.grid_world import GridWorld, Agent
    from core.sound_source import SoundSource, Wall
    from utils.audio_processing import get_audio_observation_features


class PygameVisualizer:
    """
    Pygame-based visualizer for the sound navigation environment.
    """
    
    def __init__(self, grid_width=25, grid_height=25, cell_size=20):
        """
        Initialize the Pygame visualizer.
        
        Args:
            grid_width: Width of the grid
            grid_height: Height of the grid
            cell_size: Size of each cell in pixels
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.width = grid_width * cell_size
        self.height = grid_height * cell_size
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Sound Navigation Environment")
        self.clock = pygame.time.Clock()
        
        # Define colors
        self.colors = {
            'empty': (255, 255, 255),      # White
            'wall': (100, 100, 100),       # Gray
            'agent': (0, 0, 255),          # Blue
            'sound_source': (255, 0, 0),   # Red
            'visited': (200, 200, 255)     # Light blue
        }
        
        # Track visited cells
        self.visited_cells = set()
    
    def draw_grid(self, env):
        """
        Draw the grid environment.
        
        Args:
            env: Task environment to visualize
        """
        # Clear screen
        self.screen.fill(self.colors['empty'])
        
        # Draw each cell
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                  self.cell_size, self.cell_size)
                
                # Determine cell type
                cell_type = env.grid_world.grid[x][y]
                
                if (x, y) in self.visited_cells:
                    color = self.colors['visited']
                elif cell_type == 1:  # Wall
                    color = self.colors['wall']
                else:  # Empty space
                    color = self.colors['empty']
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Border
        
        # Draw sound sources
        for source in env.grid_world.sound_sources:
            x, y = source.x, source.y
            center = (y * self.cell_size + self.cell_size // 2, 
                     x * self.cell_size + self.cell_size // 2)
            radius = self.cell_size // 3
            pygame.draw.circle(self.screen, self.colors['sound_source'], center, radius)
        
        # Draw agent
        agent_x, agent_y = env.agent.get_position()
        center = (agent_y * self.cell_size + self.cell_size // 2, 
                 agent_x * self.cell_size + self.cell_size // 2)
        radius = self.cell_size // 4
        pygame.draw.circle(self.screen, self.colors['agent'], center, radius)
        
        # Update visited cells
        self.visited_cells.add((agent_x, agent_y))
    
    def update(self, env):
        """
        Update the visualization with the current environment state.
        
        Args:
            env: Current environment state
        """
        self.draw_grid(env)
        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS
    
    def close(self):
        """
        Close the Pygame window.
        """
        pygame.quit()


def visualize_environment(env, title="Sound Navigation Environment"):
    """
    Visualize the environment using Pygame.
    
    Args:
        env: Environment to visualize
        title: Window title
    """
    # Create visualizer
    viz = PygameVisualizer()
    pygame.display.set_caption(title)
    
    # Run visualization loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Update visualization
        viz.draw_grid(env)
        pygame.display.flip()
        clock.tick(60)
    
    viz.close()


def test_visualization():
    """
    Test function to demonstrate the visualization.
    """
    from core.tasks import create_task_environment
    
    # Create a simple environment
    env = create_task_environment(task_type=1, num_sources=3)
    
    # Place some walls
    env.grid_world.place_wall(5, 5, 0.5)
    env.grid_world.place_wall(5, 6, 0.7)
    env.grid_world.place_wall(5, 7, 0.3)
    
    # Reset to initialize
    env.reset()
    
    # Run a simple test
    viz = PygameVisualizer()
    
    running = True
    step = 0
    
    while running and step < 100:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Take a random action
        action = np.random.randint(0, 5)
        obs, reward, done = env.step(action)
        
        # Update visualization
        viz.update(env)
        
        if done:
            env.reset()
        
        step += 1
    
    viz.close()


if __name__ == "__main__":
    test_visualization()