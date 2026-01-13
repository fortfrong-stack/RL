import numpy as np
import matplotlib.pyplot as plt
from core.sound_source import Wall, SoundSource, propagate_sound
from utils.audio_processing import get_audio_observation_features


class GridWorld:
    """
    A 25x25 grid world for the sound-based navigation task.
    Cell types: 0 - empty, 1 - wall, 2 - agent, 3 - sound source
    """
    
    def __init__(self, width: int = 25, height: int = 25):
        """
        Initialize the grid world.
        
        Args:
            width: Width of the grid (default 25)
            height: Height of the grid (default 25)
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        
        # Initialize agent and sound sources
        self.agent_pos = None
        self.sound_sources = []
        self.wall_objects = []  # Store Wall objects instead of just coordinates
        
    def reset(self) -> None:
        """Reset the grid to initial state"""
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.agent_pos = None
        self.sound_sources = []
        self.wall_objects = []
        
    def get_state(self) -> np.ndarray:
        """
        Return the current state of the grid.
        
        Returns:
            Copy of the grid state
        """
        return self.grid.copy()
        
    def render(self) -> None:
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
        
    def is_valid_position(self, x: int, y: int) -> bool:
        """
        Check if position is within bounds.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if position is valid, False otherwise
        """
        return 0 <= x < self.width and 0 <= y < self.height
        
    def place_wall(self, x: int, y: int, permeability: float = 0.5) -> None:
        """
        Place a wall at the given position with permeability.
        
        Args:
            x: X coordinate
            y: Y coordinate
            permeability: Permeability of the wall (default 0.5)
        """
        if self.is_valid_position(x, y):
            self.grid[x][y] = 1
            wall = Wall(x, y, permeability)
            self.wall_objects.append(wall)
            
    def place_sound_source(self, sound_source: 'SoundSource') -> None:
        """
        Place a sound source on the grid.
        
        Args:
            sound_source: SoundSource object to place
        """
        if self.is_valid_position(sound_source.x, sound_source.y):
            self.sound_sources.append(sound_source)
            
    def place_agent(self, x: int, y: int) -> bool:
        """
        Place the agent at the given position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if placement was successful, False otherwise
        """
        if self.is_valid_position(x, y) and self.grid[x][y] == 0:
            self.agent_pos = (x, y)
            return True
        return False
    
    def compute_sound_map(self) -> np.ndarray:
        """
        Compute the sound propagation map based on sources and walls.
        
        Returns:
            2D numpy array representing sound intensity at each cell
        """
        return propagate_sound(self.grid, self.sound_sources, self.wall_objects)


class Agent:
    """
    Basic agent that can move in the grid world.
    Actions: up, down, left, right, stay
    """
    
    def __init__(self, start_x: int = 0, start_y: int = 0):
        """
        Initialize the agent.
        
        Args:
            start_x: Starting X coordinate (default 0)
            start_y: Starting Y coordinate (default 0)
        """
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
        
    def move(self, action: int, grid_world: 'GridWorld') -> tuple:
        """
        Move the agent according to the action in the given grid world.
        
        Args:
            action: Action index (0-4) or action name ('up', 'down', etc.)
            grid_world: GridWorld instance to move in
            
        Returns:
            New position of the agent as (x, y) tuple
        """
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
        
    def get_position(self) -> tuple:
        """
        Get current position of the agent.
        
        Returns:
            Current position as (x, y) tuple
        """
        return self.position
        
    def observe(self, sound_map: np.ndarray = None, grid_world: 'GridWorld' = None) -> np.ndarray:
        """
        Audio observation for the agent.
        Returns audio features extracted from the sound at the agent's position.
        
        Args:
            sound_map: Sound intensity map (optional)
            grid_world: GridWorld instance (optional)
            
        Returns:
            Audio observation features as numpy array
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

    def _get_frequency_content_at_position(self, grid_world: 'GridWorld') -> float:
        """
        Get the dominant frequency content at the agent's position based on nearby sources.
        This considers both the distance and volume of sources to determine which one is loudest.
        
        Args:
            grid_world: GridWorld instance to analyze
            
        Returns:
            Dominant frequency content (0.0-1.0)
        """
        if grid_world is None or not grid_world.sound_sources:
            return 0.5  # Default frequency content

        # Find the most prominent sound source at the agent's position based on perceived loudness
        max_perceived_loudness = -1
        dominant_frequency = 0.5
        
        for source in grid_world.sound_sources:
            # Calculate Manhattan distance
            distance = abs(self.x - source.x) + abs(self.y - source.y)
            
            # Calculate perceived loudness based on distance and source volume
            # Using inverse relationship similar to our sound propagation
            perceived_loudness = source.volume / (1 + 0.5 * distance + 0.1 * distance**1.5)
            
            if perceived_loudness > max_perceived_loudness:
                max_perceived_loudness = perceived_loudness
                dominant_frequency = source.frequency
                
        return dominant_frequency
