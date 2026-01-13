import random
import numpy as np


class SoundSource:
    """
    Class representing a sound source in the grid world.
    """
    
    def __init__(self, x, y, volume=0.5, frequency=0.5):
        self.x = x
        self.y = y
        self.volume = volume  # Determines the radius of sound propagation (0.1-1.0)
        self.frequency = frequency  # Probability of emitting sound per step (0.1-1.0)
        
    def emit_sound(self):
        """
        Generate sound signal with given probability based on frequency.
        Returns True if sound is emitted, False otherwise.
        """
        return random.random() < self.frequency


class Wall:
    """
    Class representing a wall in the grid world with permeability properties.
    """
    
    def __init__(self, x, y, permeability=0.5):
        self.x = x
        self.y = y
        self.permeability = max(0.25, min(1.0, permeability))  # Clamp between 0.25 and 1.0
        
    def is_passable_by_agent(self):
        """
        Check if the wall is passable by the agent.
        Currently, walls are not passable by agents (this could be changed in future).
        """
        return False


def propagate_sound(grid, sources, walls):
    """
    Propagate sound through the grid with more realistic physics.
    
    Args:
        grid: The grid world (for checking obstacles)
        sources: List of SoundSource objects
        walls: List of Wall objects
    
    Returns:
        sound_map: 2D numpy array representing sound intensity at each cell
    """
    sound_map = np.zeros_like(grid, dtype=float)
    height, width = grid.shape
    
    for source in sources:
        if source.emit_sound():
            # Use a more realistic propagation considering distance from source
            max_distance = int(source.volume * 20)  # Adjust max distance based on volume
            
            # Instead of BFS, calculate intensity based on distance from source
            for x in range(height):
                for y in range(width):
                    # Skip if it's the source position (will have max volume)
                    if x == source.x and y == source.y:
                        sound_map[x][y] += source.volume
                        continue
                    
                    # Calculate Manhattan distance from source
                    distance = abs(x - source.x) + abs(y - source.y)
                    
                    # Skip if beyond max propagation distance
                    if distance > max_distance:
                        continue
                    
                    # Calculate base intensity using inverse square law approximation
                    # In 2D grid, we use inverse law (between inverse and inverse square)
                    base_attenuation = source.volume / (1 + 0.5 * distance + 0.1 * distance**1.5)
                    
                    # Check if there's a direct path affected by walls
                    path_attenuation = calculate_path_attenuation(
                        source.x, source.y, x, y, grid, walls, source.frequency
                    )
                    
                    # Final intensity is base attenuation modified by path effects
                    intensity = base_attenuation * path_attenuation
                    
                    # Only add if above threshold
                    if intensity > 0.01:
                        sound_map[x][y] += intensity
    
    return sound_map


def calculate_path_attenuation(start_x, start_y, end_x, end_y, grid, walls, frequency):
    """
    Calculate attenuation along the path from source to target considering obstacles.
    Uses a simplified ray-tracing approach to account for obstacles in the path.
    
    Args:
        start_x, start_y: Source position
        end_x, end_y: Target position
        grid: Grid world
        walls: List of Wall objects
        frequency: Frequency of the sound (for frequency-dependent attenuation)
    
    Returns:
        Attenuation factor (0.0 to 1.0)
    """
    # For now, use a simple approach that checks if there's a wall directly between source and target
    # This is a simplified ray-casting approach
    dx = end_x - start_x
    dy = end_y - start_y
    
    # Determine the direction and number of steps
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return 1.0  # Same position
    
    # Calculate increments for each step
    x_inc = dx / steps if steps != 0 else 0
    y_inc = dy / steps if steps != 0 else 0
    
    current_attenuation = 1.0
    
    # Move along the path and apply attenuation for each obstacle encountered
    for i in range(1, int(steps) + 1):
        curr_x = int(round(start_x + x_inc * i))
        curr_y = int(round(start_y + y_inc * i))
        
        # Check if this position is out of bounds
        if not (0 <= curr_x < grid.shape[0] and 0 <= curr_y < grid.shape[1]):
            continue
        
        # Check if this position is a wall in the grid
        if grid[curr_x][curr_y] == 1:
            # Find the corresponding wall object to get permeability
            wall_permeability = 0.5  # Default permeability
            for wall in walls:
                if wall.x == curr_x and wall.y == curr_y:
                    wall_permeability = wall.permeability
                    break
            
            # Apply frequency-dependent attenuation
            # Lower frequencies penetrate better than higher frequencies
            freq_factor = 0.7 + 0.3 * (1.0 - frequency)  # Lower freq = less attenuation
            current_attenuation *= wall_permeability * freq_factor
    
    return current_attenuation
