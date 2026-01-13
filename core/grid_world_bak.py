import numpy as np
import matplotlib.pyplot as plt
from core.sound_source import Wall, SoundSource, propagate_sound
from utils.audio_processing import get_audio_observation_features


class GridWorld:
    """
    A 25x25 grid world for the sound-based navigation task.
    Cell types: 0 - empty, 1 - wall, 2 - agent, 3 - sound source
    """
    
    def __init__(self, width=25, height=25):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        
        # Initialize agent and sound sources
        self.agent_pos = None
        self.sound_sources = []
        self.wall_objects = []  # Store Wall objects instead of just coordinates
        
    def reset(self):
        """Reset the grid to initial state"""
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.agent_pos = None
        self.sound_sources = []
        self.wall_objects = []
        
    def get_state(self):
        """Return the current state of the grid"""
        return self.grid.copy()
        
    def render(self):
        """Visualize the current state of the grid"""
        plt.figure(figsize=(8, 8))
        
        # Create a copy of the grid for visualization
        vis_grid = self.grid.copy()
        
        # Mark the agent position if exists
        if self.agent_pos is not None:
            x, y = self.agent_pos
            vis_grid[x][y] = 2
            
        # Mark sound sources if exist
        for source in self.sound_sources:
            vis_grid[source.x][source.y] = 3
            
        plt.imshow(vis_grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Cell Type (0: Empty, 1: Wall, 2: Agent, 3: Sound Source)')
        plt.title('Grid World Visualization')
        plt.show()
        
    def is_valid_position(self, x, y):
        """Check if position is within bounds"""
        return 0 <= x < self.width and 0 <= y < self.height
        
    def place_wall(self, x, y, permeability=0.5):
        """Place a wall at the given position with permeability"""
        if self.is_valid_position(x, y):
            self.grid[x][y] = 1
            wall = Wall(x, y, permeability)
            self.wall_objects.append(wall)
            
    def place_sound_source(self, sound_source):
        """Place a sound source on the grid"""
        if self.is_valid_position(sound_source.x, sound_source.y):
            self.sound_sources.append(sound_source)
            
    def place_agent(self, x, y):
        """Place the agent at the given position"""
        if self.is_valid_position(x, y) and self.grid[x][y] == 0:
            self.agent_pos = (x, y)
            return True
        return False
    
    def compute_sound_map(self):
        """Compute the sound propagation map based on sources and walls"""
        return propagate_sound(self.grid, self.sound_sources, self.wall_objects)


class Agent:
    """
    Basic agent that can move in the grid world.
    Actions: up, down, left, right, stay
    """
    
    def __init__(self, start_x=0, start_y=0):
        self.x = start_x
        self.y = start_y
        self.position = (start_x, start_y)
        
        # Define possible actions
        self.actions = {
            0: 'up',
            1: 'down', 
            2: 'left',
            3: 'right',
            4: 'stay'
        }
        
        # Direction vectors for movement
        self.action_vectors = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
            'stay': (0, 0)
        }
        
    def move(self, action, grid_world):
        """Move the agent according to the action in the given grid world"""
        # Handle numpy integers as well as Python integers
        import numbers
        if isinstance(action, numbers.Integral):
            action = self.actions[action]
            
        if action not in self.action_vectors:
            raise ValueError(f"Invalid action: {action}")
            
        dx, dy = self.action_vectors[action]
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check boundaries and collisions with walls
        # We need to check the actual grid state, not a copy
        if (grid_world.is_valid_position(new_x, new_y) and 
            grid_world.grid[new_x][new_y] != 1):  # Not a wall
            self.x = new_x
            self.y = new_y
            self.position = (self.x, self.y)
            
            # Update the agent position in the grid world
            grid_world.agent_pos = (self.x, self.y)
            
        return self.position
        
    def get_position(self):
        """Get current position of the agent"""
        return self.position
        
    def observe(self, sound_map=None, grid_world=None):
        """
        Audio observation for the agent.
        Returns audio features extracted from the sound at the agent's position.
        """
        if sound_map is not None:
            intensity = sound_map[self.x][self.y]
            
            # Determine frequency content from nearby sound sources
            frequency_content = self._get_frequency_content_at_position(grid_world)
            
            # Get audio observation features
            return get_audio_observation_features(intensity, frequency_content)
        else:
            # Return default audio features when no sound map is available
            return get_audio_observation_features(0.0, 0.5)  # Default: no intensity, medium frequency

    def _get_frequency_content_at_position(self, grid_world):
        """
        Get the dominant frequency content at the agent's position based on nearby sources.
        This is a simplified approach - in a real implementation, we might consider
        the contribution of multiple sources based on their distance and volume.
        """
        if grid_world is None or not grid_world.sound_sources:
            return 0.5  # Default frequency content
        
        # Find the closest sound source to determine frequency content
        min_distance = float('inf')
        closest_source_freq = 0.5
        
        for source in grid_world.sound_sources:
            distance = abs(self.x - source.x) + abs(self.y - source.y)  # Manhattan distance
            if distance < min_distance:
                min_distance = distance
                closest_source_freq = source.frequency
                
        return closest_source_freq