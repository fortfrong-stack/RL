"""
Enhanced implementation of the three tasks for Stage 4 with improved reward systems:
1. Finding all sound sources
2. Finding the quietest place
3. Following a moving sound source

Improvements:
- Normalized rewards for better stability
- Increased step penalties to encourage efficiency
- Added intermediate rewards for progress
- Used potential functions for smoother rewards
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    # When imported as part of the package
    from .grid_world import GridWorld, Agent
    from .sound_source import SoundSource
except ImportError:
    # When run directly
    from core.grid_world import GridWorld, Agent
    from core.sound_source import SoundSource


class EnhancedTaskEnvironment:
    """
    Base class for enhanced task environments that extends GridWorld with improved reward system and termination conditions.
    """
    
    def __init__(self, width=25, height=25, task_type=1):
        self.width = width
        self.height = height
        self.task_type = task_type
        self.grid_world = GridWorld(width, height)
        self.agent = None
        self.current_step = 0
        self.max_steps = 500
        self.done = False
        self.total_reward = 0
        self.found_sources = set()
        
    def reset(self):
        """Reset the environment to initial state."""
        self.grid_world.reset()
        self.current_step = 0
        self.done = False
        self.total_reward = 0
        self.found_sources = set()
        return self.get_observation()

    def step(self, action):
        """Execute an action and return (observation, reward, done)."""
        if self.done:
            return self.get_observation(), 0, True
            
        old_pos = self.agent.get_position()
        
        # Move the agent
        new_pos = self.agent.move(action, self.grid_world)
        
        # Compute sound map after movement
        sound_map = self.grid_world.compute_sound_map()
        
        # Calculate reward based on task type
        reward = self.calculate_reward(old_pos, new_pos, sound_map)
        
        # Update found sources if applicable
        self.update_found_sources()
        
        # Check if task is completed
        self.check_termination(sound_map)
        
        self.current_step += 1
        
        # Check if max steps reached
        if self.current_step >= self.max_steps:
            self.done = True
            
        return self.get_observation(sound_map), reward, self.done
    
    def get_observation(self, sound_map=None):
        """Get current observation for the agent."""
        if sound_map is None:
            sound_map = self.grid_world.compute_sound_map()
        return self.agent.observe(sound_map=sound_map, grid_world=self.grid_world)
    
    def calculate_reward(self, old_pos, new_pos, sound_map):
        """Calculate reward based on the specific task."""
        raise NotImplementedError("Subclasses must implement calculate_reward")
        
    def check_termination(self, sound_map):
        """Check if the task is completed."""
        raise NotImplementedError("Subclasses must implement check_termination")
        
    def update_found_sources(self):
        """Update tracking of found sources."""
        agent_x, agent_y = self.agent.get_position()
        current_pos = (agent_x, agent_y)
        
        # Check if agent is on a sound source
        for source in self.grid_world.sound_sources:
            if (source.x, source.y) == current_pos:
                self.found_sources.add(current_pos)


class EnhancedFindAllSourcesTask(EnhancedTaskEnvironment):
    """
    Enhanced Task 1: Find all sound sources
    Improvements:
    - Increased step penalty to encourage efficiency
    - Added intermediate rewards for approaching sources
    - Used potential function based on distance to nearest unfound source
    - Normalized rewards to [-1, 1] range
    """
    
    def __init__(self, width=25, height=25, num_sources=None):
        super().__init__(width, height, task_type=1)
        self.num_sources = num_sources or np.random.randint(1, 6)
        self.initial_sources_count = self.num_sources
        self.found_sources = set()
        self.last_distances_to_sources = {}
    
    def reset(self):
        """Reset the environment to initial state."""
        self.grid_world.reset()
        self.current_step = 0
        self.done = False
        self.total_reward = 0
        self.found_sources = set()
        self.last_distances_to_sources = {}

        # Place agent if not already placed
        if self.agent is None:
            placed = False
            while not placed:
                x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
                if self.grid_world.grid[x][y] == 0:  # Empty space
                    self.agent = Agent(x, y)
                    self.grid_world.place_agent(x, y)
                    placed = True
        
        # Place sound sources randomly
        for _ in range(self.num_sources):
            placed = False
            while not placed:
                x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
                if (self.grid_world.grid[x][y] == 0 and  # Empty space
                    (self.agent is None or (x, y) != self.agent.get_position())):  # Not on agent
                    source = SoundSource(x, y, 
                                       volume=np.random.uniform(0.3, 1.0), 
                                       frequency=np.random.uniform(0.2, 1.0))
                    self.grid_world.place_sound_source(source)
                    placed = True
        
        # Initialize distances to unfound sources
        for source in self.grid_world.sound_sources:
            source_pos = (source.x, source.y)
            if source_pos not in self.found_sources:
                agent_x, agent_y = self.agent.get_position()
                dist = abs(agent_x - source.x) + abs(agent_y - source.y)
                self.last_distances_to_sources[source_pos] = dist
                
        return self.get_observation()
        
    def calculate_reward(self, old_pos, new_pos, sound_map):
        """Calculate enhanced reward for finding all sources task."""
        # Base step penalty to encourage efficiency
        base_penalty = -0.5  # Increased from -0.1
        
        # Check if agent is now on a source that wasn't previously found
        agent_x, agent_y = new_pos
        current_pos = (agent_x, agent_y)
        
        reward = base_penalty  # Start with step penalty
        
        # Reward for finding new sources
        newly_found = False
        for source in self.grid_world.sound_sources:
            if (source.x, source.y) == current_pos and current_pos not in self.found_sources:
                reward += 20.0  # Increased reward for finding a new source
                self.found_sources.add(current_pos)
                newly_found = True
                # Remove from distance tracking since it's found
                if current_pos in self.last_distances_to_sources:
                    del self.last_distances_to_sources[current_pos]
                
        # Additional reward if all sources are found
        if len(self.found_sources) == len(self.grid_world.sound_sources):
            reward += 100.0  # Increased bonus for finding all sources
            return reward  # Early return for completion bonus
            
        # Intermediate reward for getting closer to unfound sources
        if not newly_found:
            for source in self.grid_world.sound_sources:
                source_pos = (source.x, source.y)
                if source_pos not in self.found_sources:  # Only consider unfound sources
                    # Calculate new distance
                    new_dist = abs(agent_x - source.x) + abs(agent_y - source.y)
                    
                    # Get previous distance if available
                    if source_pos in self.last_distances_to_sources:
                        old_dist = self.last_distances_to_sources[source_pos]
                        
                        # Reward for getting closer, penalty for getting farther
                        if new_dist < old_dist:
                            # Reward based on improvement, normalized
                            improvement = (old_dist - new_dist) / max(self.width, self.height)
                            reward += 2.0 * improvement  # Reward for progress toward unfound source
                        elif new_dist > old_dist:
                            # Penalty for moving away, normalized
                            deterioration = (new_dist - old_dist) / max(self.width, self.height)
                            reward -= 1.0 * deterioration  # Penalty for moving away
                    
                    # Update stored distance
                    self.last_distances_to_sources[source_pos] = new_dist
        
        # Normalize reward to prevent extreme values
        reward = np.clip(reward, -10.0, 100.0)
            
        return reward
    
    def check_termination(self, sound_map):
        """Check if all sources have been found."""
        if len(self.found_sources) == len(self.grid_world.sound_sources):
            self.done = True


class EnhancedFindQuietestPlaceTask(EnhancedTaskEnvironment):
    """
    Enhanced Task 2: Find the quietest place
    Improvements:
    - Normalized intensity-based rewards
    - Added potential-based reward using distance to quietest cell
    - Smoothed reward function to reduce instability
    - Added intermediate rewards for approaching quiet areas
    """
    
    def __init__(self, width=25, height=25, num_sources=None):
        super().__init__(width, height, task_type=2)
        self.num_sources = num_sources or np.random.randint(1, 6)
        self.sound_map = None
        self.quietest_cell = None
        self.min_intensity = float('inf')
        self.last_agent_pos = None
        self.last_intensity = None
    
    def reset(self):
        """Reset the environment to initial state."""
        self.grid_world.reset()
        self.current_step = 0
        self.done = False
        self.total_reward = 0
        self.found_sources = set()
        self.last_agent_pos = None
        self.last_intensity = None

        # Place agent if not already placed
        if self.agent is None:
            placed = False
            while not placed:
                x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
                if self.grid_world.grid[x][y] == 0:  # Empty space
                    self.agent = Agent(x, y)
                    self.grid_world.place_agent(x, y)
                    placed = True
        
        # Place sound sources randomly after agent placement
        for _ in range(self.num_sources):
            placed = False
            while not placed:
                x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
                if (self.grid_world.grid[x][y] == 0 and  # Empty space
                    (x, y) != self.agent.get_position()):  # Not on agent
                    source = SoundSource(x, y, 
                                       volume=np.random.uniform(0.3, 1.0), 
                                       frequency=np.random.uniform(0.2, 1.0))
                    self.grid_world.place_sound_source(source)
                    placed = True
        
        # Compute sound map and find quietest cell immediately
        self.sound_map = self.grid_world.compute_sound_map()
        self.min_intensity = np.min(self.sound_map)
        quietest_positions = np.where(self.sound_map == self.min_intensity)
        if len(quietest_positions[0]) > 0:
            # Pick the first quietest position
            self.quietest_cell = (quietest_positions[0][0], quietest_positions[1][0])
        
        self.last_agent_pos = self.agent.get_position()
        agent_x, agent_y = self.last_agent_pos
        self.last_intensity = self.sound_map[agent_x][agent_y]
        
        return self.get_observation(self.sound_map)
        
    def calculate_reward(self, old_pos, new_pos, sound_map):
        """Calculate enhanced reward for finding quietest place task."""
        self.sound_map = sound_map
        
        agent_x, agent_y = new_pos
        current_intensity = sound_map[agent_x][agent_y]
        
        # Normalize reward based on relative improvement
        # Instead of using absolute intensity, compare to global min/max
        normalized_improvement = 0
        if self.min_intensity != np.max(sound_map):  # Avoid division by zero
            # Calculate how much closer we are to the minimum
            current_diff = current_intensity - self.min_intensity
            max_possible_diff = np.max(sound_map) - self.min_intensity
            
            if max_possible_diff > 0:
                # Higher reward when we're closer to the minimum
                normalized_improvement = 1.0 - (current_diff / max_possible_diff)
        
        # Base reward based on improvement in quietness
        reward = normalized_improvement * 5.0  # Scale to reasonable range
        
        # Additional reward based on comparison with previous position
        if self.last_intensity is not None:
            if current_intensity < self.last_intensity:
                # Reward for finding quieter spot
                improvement_ratio = (self.last_intensity - current_intensity) / self.last_intensity
                reward += 3.0 * improvement_ratio
            elif current_intensity > self.last_intensity:
                # Smaller penalty for going to louder area
                deterioration_ratio = (current_intensity - self.last_intensity) / current_intensity if current_intensity > 0 else 0
                reward -= 1.5 * deterioration_ratio
        
        # Distance-based potential reward toward quietest cell
        if self.quietest_cell:
            # Calculate Manhattan distance to quietest cell
            current_dist = abs(agent_x - self.quietest_cell[0]) + abs(agent_y - self.quietest_cell[1])
            
            # If we know previous position, give reward for getting closer
            if self.last_agent_pos:
                last_dist = abs(self.last_agent_pos[0] - self.quietest_cell[0]) + abs(self.last_agent_pos[1] - self.quietest_cell[1])
                
                if current_dist < last_dist:
                    # Reward for getting closer to target
                    dist_improvement = (last_dist - current_dist) / max(self.width, self.height)
                    reward += 2.0 * dist_improvement
                elif current_dist > last_dist:
                    # Small penalty for moving away
                    dist_penalty = (current_dist - last_dist) / max(self.width, self.height)
                    reward -= 1.0 * dist_penalty
        
        # Bonus for reaching the quietest spot
        if self.quietest_cell and new_pos == self.quietest_cell:
            reward += 50.0  # Bonus for reaching the quietest spot
        
        # Update tracking
        self.last_agent_pos = new_pos
        self.last_intensity = current_intensity
        
        # Normalize reward to prevent extreme values
        reward = np.clip(reward, -10.0, 50.0)
            
        return reward
    
    def check_termination(self, sound_map):
        """Check if agent is at the quietest place."""
        if self.quietest_cell:
            agent_x, agent_y = self.agent.get_position()
            if (agent_x, agent_y) == self.quietest_cell:
                self.done = True


class EnhancedFollowMovingSourceTask(EnhancedTaskEnvironment):
    """
    Enhanced Task 3: Follow a moving sound source
    Improvements:
    - Smoother distance-based rewards
    - Potential function to guide toward source
    - Reduced discreteness of distance changes
    - Added velocity prediction component
    """
    
    def __init__(self, width=25, height=25, move_interval=10):
        super().__init__(width, height, task_type=3)
        self.move_interval = move_interval  # How often the source moves
        self.source_last_pos = None
        self.distance_history = []
        self.velocity_estimate = None
        self.predicted_next_pos = None
    
    def reset(self):
        """Reset the environment to initial state."""
        # First call parent reset but don't return the observation yet
        # because agent doesn't exist yet
        self.grid_world.reset()
        self.current_step = 0
        self.done = False
        self.total_reward = 0
        self.found_sources = set()
        self.velocity_estimate = None
        self.predicted_next_pos = None

        # Place a single moving source
        placed = False
        while not placed:
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            if self.grid_world.grid[x][y] == 0:  # Empty space
                source = SoundSource(x, y, volume=0.8, frequency=0.6)
                self.grid_world.place_sound_source(source)
                self.source_last_pos = (x, y)
                placed = True
        
        # Place agent
        placed = False
        while not placed:
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            if self.grid_world.grid[x][y] == 0:  # Empty space
                self.agent = Agent(x, y)
                self.grid_world.place_agent(x, y)
                placed = True
                
        # Now return the observation with the agent in place
        return self.get_observation()
    
    def calculate_reward(self, old_pos, new_pos, sound_map):
        """Calculate enhanced reward for following moving source task."""
        agent_x, agent_y = new_pos
        
        # Get current source position
        if self.grid_world.sound_sources:
            source = self.grid_world.sound_sources[0]  # Only one source in this task
            source_x, source_y = source.x, source.y
        else:
            return 0  # No source to follow
        
        # Calculate distance to source
        current_distance = abs(agent_x - source_x) + abs(agent_y - source_y)
        
        # Calculate previous distance if available
        reward = -0.3  # Small step penalty to encourage efficiency
        
        if len(self.distance_history) > 0:
            prev_distance = self.distance_history[-1]
            
            # Calculate smooth reward based on distance change
            # Instead of discrete +5/-5, use continuous function
            distance_change = prev_distance - current_distance  # Positive if getting closer
            reward += 3.0 * distance_change / max(self.width, self.height)  # Normalize
            
            # Additional reward for being close
            if current_distance <= 3:
                reward += (4 - current_distance) * 0.5  # Extra reward for being very close
        else:
            # For first step, just store the distance
            pass
        
        # Add to distance history
        self.distance_history.append(current_distance)
        
        # Check if agent caught the source (distance < 2)
        if current_distance < 2:
            reward += 100.0  # Big reward for catching the source
            
        # Potential field reward: reward for moving toward estimated source position
        if self.predicted_next_pos:
            # Calculate distance to predicted position
            pred_dist = abs(agent_x - self.predicted_next_pos[0]) + abs(agent_y - self.predicted_next_pos[1])
            # Encourage moving toward predicted position
            reward += 1.0 / (pred_dist + 1)  # Higher reward for being closer to predicted position
        
        # Update velocity estimate if we have previous positions
        if self.source_last_pos and self.source_last_pos != (source_x, source_y):
            # Source moved, so we can estimate velocity
            dx = source_x - self.source_last_pos[0]
            dy = source_y - self.source_last_pos[1]
            self.velocity_estimate = (dx, dy)
            
            # Predict next position based on velocity
            self.predicted_next_pos = (source_x + dx, source_y + dy)
        
        # Update last source position
        self.source_last_pos = (source_x, source_y)
        
        # Normalize reward to prevent extreme values
        reward = np.clip(reward, -10.0, 100.0)
            
        return reward
    
    def check_termination(self, sound_map):
        """Check if agent caught the source."""
        if self.grid_world.sound_sources:
            source = self.grid_world.sound_sources[0]
            agent_x, agent_y = self.agent.get_position()
            distance = abs(agent_x - source.x) + abs(agent_y - source.y)
            
            if distance < 2:  # Close enough to "catch" the source
                self.done = True
    
    def step(self, action):
        """Execute an action and move the source periodically."""
        # Move the source randomly every move_interval steps
        if self.current_step > 0 and self.current_step % self.move_interval == 0:
            self._move_source_randomly()
            
        return super().step(action)
    
    def _move_source_randomly(self):
        """Move the sound source to a new random valid position."""
        if not self.grid_world.sound_sources:
            return
            
        source = self.grid_world.sound_sources[0]
        
        # Try to find a new valid position for the source
        attempts = 0
        while attempts < 10:  # Limit attempts to avoid infinite loops
            new_x = np.clip(source.x + np.random.randint(-3, 4), 0, self.width - 1)
            new_y = np.clip(source.y + np.random.randint(-3, 4), 0, self.height - 1)
            
            # Check if the new position is valid (empty and not occupied by agent)
            if (self.grid_world.is_valid_position(new_x, new_y) and 
                self.grid_world.grid[new_x][new_y] == 0 and
                (new_x, new_y) != self.agent.get_position()):
                
                # Update source position
                source.x = new_x
                source.y = new_y
                break
                
            attempts += 1


def create_enhanced_task_environment(task_type, **kwargs):
    """
    Factory function to create the appropriate enhanced task environment.
    
    Args:
        task_type: 1 for find all sources, 2 for quietest place, 3 for follow moving source
        **kwargs: Additional arguments passed to the task constructor
    
    Returns:
        An instance of the appropriate enhanced task environment
    """
    if task_type == 1:
        return EnhancedFindAllSourcesTask(**kwargs)
    elif task_type == 2:
        return EnhancedFindQuietestPlaceTask(**kwargs)
    elif task_type == 3:
        return EnhancedFollowMovingSourceTask(**kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")