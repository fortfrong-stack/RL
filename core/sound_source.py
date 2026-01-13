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
    Propagate sound through the grid with wavefront propagation algorithm.
    
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
            # Start propagation from the source position
            max_distance = int(source.volume * 20)  # Adjust max distance based on volume
            
            # Initialize queue for BFS-like propagation
            queue = [(source.x, source.y, source.volume)]
            visited = set([(source.x, source.y)])  # Track visited cells to avoid duplicates
            
            while queue:
                x, y, intensity = queue.pop(0)
                
                # Check bounds and minimum intensity
                if not (0 <= x < height and 0 <= y < width and intensity > 0.01):
                    continue
                
                # Add sound intensity to this cell
                current_intensity = sound_map[x][y]
                sound_map[x][y] = current_intensity + intensity
                
                # Continue propagation to neighbors
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4 directions
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= nx < height and 0 <= ny < width and 
                        (nx, ny) not in visited and grid[nx][ny] != 1):  # Not a wall in grid
                        
                        # Calculate attenuation based on distance and permeability
                        new_intensity = intensity * 0.9  # Base attenuation
                        
                        # Apply wall permeability if neighbor is a wall object
                        for wall in walls:
                            if wall.x == nx and wall.y == ny:
                                new_intensity *= wall.permeability
                                break
                                
                        if new_intensity > 0.01:  # Only continue if significant
                            queue.append((nx, ny, new_intensity))
                            visited.add((nx, ny))
    
    # Apply wall permeability effects to sound map
    for wall in walls:
        if 0 <= wall.x < height and 0 <= wall.y < width:
            sound_map[wall.x][wall.y] *= wall.permeability
    
    return sound_map