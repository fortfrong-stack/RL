"""
Implementation of the three tasks for Stage 4:
1. Finding all sound sources
2. Finding the quietest place
3. Following a moving sound source
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


class TaskEnvironment:
    """
    Base class for task environments that extends GridWorld with reward system and termination conditions.
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


class FindAllSourcesTask(TaskEnvironment):
    """
    Task 1: Find all sound sources
    - Parameters: 1-5 sources
    - Reward system: +10 for finding new source, -0.1 per step, +50 for finding all
    - Termination: all sources found or max steps
    """
    
    def __init__(self, width=25, height=25, num_sources=None):
        super().__init__(width, height, task_type=1)
        self.num_sources = num_sources or np.random.randint(1, 6)
        self.initial_sources_count = self.num_sources
        self.found_sources = set()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.grid_world.reset()
        self.current_step = 0
        self.done = False
        self.total_reward = 0
        self.found_sources = set()
        
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
        
        return self.get_observation()
        
    def calculate_reward(self, old_pos, new_pos, sound_map):
        """Calculate reward for finding all sources task."""
        reward = -0.1  # Small penalty for each step to encourage efficiency
        
        # Check if agent is now on a source that wasn't previously found
        agent_x, agent_y = new_pos
        current_pos = (agent_x, agent_y)
        
        for source in self.grid_world.sound_sources:
            if (source.x, source.y) == current_pos and current_pos not in self.found_sources:
                reward += 10  # Reward for finding a new source
                self.found_sources.add(current_pos)
                
        # Additional reward if all sources are found
        if len(self.found_sources) == len(self.grid_world.sound_sources):
            reward += 50  # Large bonus for finding all sources
            
        return reward
    
    def check_termination(self, sound_map):
        """Check if all sources have been found."""
        if len(self.found_sources) == len(self.grid_world.sound_sources):
            self.done = True


class FindQuietestPlaceTask(TaskEnvironment):
    """
    Task 2: Find the quietest place
    - Parameters: 1-5 sources
    - Reward system: -1 * current intensity, +100 for reaching quietest spot
    - Termination: agent in quietest spot or max steps
    """
    
    def __init__(self, width=25, height=25, num_sources=None):
        super().__init__(width, height, task_type=2)
        self.num_sources = num_sources or np.random.randint(1, 6)
        self.sound_map = None
        self.quietest_cell = None
        self.min_intensity = float('inf')
    
    def reset(self):
        """Reset the environment to initial state."""
        self.grid_world.reset()
        self.current_step = 0
        self.done = False
        self.total_reward = 0
        self.found_sources = set()
        
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
        
        return self.get_observation(self.sound_map)
        
    def calculate_reward(self, old_pos, new_pos, sound_map):
        """Calculate reward for finding quietest place task."""
        self.sound_map = sound_map
        
        agent_x, agent_y = new_pos
        current_intensity = sound_map[agent_x][agent_y]
        
        # Calculate reward as negative of current intensity
        reward = -current_intensity
        
        # Check if agent is at the quietest cell
        if self.quietest_cell and new_pos == self.quietest_cell:
            reward += 100  # Bonus for reaching the quietest spot
            
        return reward
    
    def check_termination(self, sound_map):
        """Check if agent is at the quietest place."""
        if self.quietest_cell:
            agent_x, agent_y = self.agent.get_position()
            if (agent_x, agent_y) == self.quietest_cell:
                self.done = True


class FollowMovingSourceTask(TaskEnvironment):
    """
    Task 3: Follow a moving sound source
    - Parameters: 1 source that moves randomly
    - Reward system: +5 for decreasing distance, -5 for increasing, +100 for catching
    - Movement: source moves randomly every N steps
    """
    
    def __init__(self, width=25, height=25, move_interval=10):
        super().__init__(width, height, task_type=3)
        self.move_interval = move_interval  # How often the source moves
        self.source_last_pos = None
        self.distance_history = []
        
    def reset(self):
        """Reset the environment to initial state."""
        # First call parent reset but don't return the observation yet
        # because agent doesn't exist yet
        self.grid_world.reset()
        self.current_step = 0
        self.done = False
        self.total_reward = 0
        self.found_sources = set()
        
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
        """Calculate reward for following moving source task."""
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
        reward = 0
        if len(self.distance_history) > 0:
            prev_distance = self.distance_history[-1]
            
            if current_distance < prev_distance:
                reward = 5  # Reward for getting closer
            elif current_distance > prev_distance:
                reward = -5  # Penalty for getting farther
            # No change if distance stays the same
        
        # Add to distance history
        self.distance_history.append(current_distance)
        
        # Check if agent caught the source (distance < 2)
        if current_distance < 2:
            reward += 100  # Big reward for catching the source
            
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
            new_x = np.clip(source.x + np.random.randint(-2, 3), 0, self.width - 1)
            new_y = np.clip(source.y + np.random.randint(-2, 3), 0, self.height - 1)
            
            # Check if the new position is valid (empty and not occupied by agent)
            if (self.grid_world.is_valid_position(new_x, new_y) and 
                self.grid_world.grid[new_x][new_y] == 0 and
                (new_x, new_y) != self.agent.get_position()):
                
                # Update source position
                source.x = new_x
                source.y = new_y
                break
                
            attempts += 1


def create_task_environment(task_type, **kwargs):
    """
    Factory function to create the appropriate task environment.
    
    Args:
        task_type: 1 for find all sources, 2 for quietest place, 3 for follow moving source
        **kwargs: Additional arguments passed to the task constructor
    
    Returns:
        An instance of the appropriate task environment
    """
    if task_type == 1:
        return FindAllSourcesTask(**kwargs)
    elif task_type == 2:
        return FindQuietestPlaceTask(**kwargs)
    elif task_type == 3:
        return FollowMovingSourceTask(**kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")