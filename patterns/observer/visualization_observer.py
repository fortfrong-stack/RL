"""
Observer pattern implementation for visualization events.
"""

from abc import ABC, abstractmethod
import pygame
import numpy as np
import sys
import os

# Add parent directory to path to import from core
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')

try:
    from core.grid_world import GridWorld, Agent
    from core.sound_source import SoundSource, Wall
    from utils.audio_processing import get_audio_observation_features
except ImportError:
    from core.grid_world import GridWorld, Agent
    from core.sound_source import SoundSource, Wall
    from utils.audio_processing import get_audio_observation_features


class Subject(ABC):
    """Abstract subject for observer pattern."""
    
    @abstractmethod
    def attach(self, observer):
        """Attach an observer."""
        pass
    
    @abstractmethod
    def detach(self, observer):
        """Detach an observer."""
        pass
    
    @abstractmethod
    def notify(self, event_type, data):
        """Notify all observers of an event."""
        pass


class VisualizationSubject(Subject):
    """Subject that notifies observers about visualization events."""
    
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        """Attach an observer."""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        """Detach an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event_type, data):
        """Notify all observers of an event."""
        for observer in self._observers:
            observer.update(event_type, data)


class Observer(ABC):
    """Abstract observer for visualization events."""
    
    @abstractmethod
    def update(self, event_type, data):
        """Update the observer with new event data."""
        pass


class VisualizationObserver(Observer):
    """Base class for visualization observers."""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
    
    def update(self, event_type, data):
        """Process visualization update."""
        pass


class AgentPositionObserver(VisualizationObserver):
    """Observer that tracks agent position changes."""
    
    def update(self, event_type, data):
        """Update when agent position changes."""
        if event_type == 'agent_move':
            self.visualizer.handle_agent_move(data['old_pos'], data['new_pos'])
        elif event_type == 'agent_reset':
            self.visualizer.handle_agent_reset(data['position'])


class SoundSourceObserver(VisualizationObserver):
    """Observer that tracks sound source changes."""
    
    def update(self, event_type, data):
        """Update when sound sources change."""
        if event_type == 'source_added':
            self.visualizer.handle_source_added(data['source'])
        elif event_type == 'source_moved':
            self.visualizer.handle_source_moved(data['old_pos'], data['new_pos'])
        elif event_type == 'source_removed':
            self.visualizer.handle_source_removed(data['source'])


class EnvironmentStateObserver(VisualizationObserver):
    """Observer that tracks overall environment state."""
    
    def update(self, event_type, data):
        """Update when environment state changes."""
        if event_type in ['env_reset', 'env_step', 'env_render']:
            self.visualizer.handle_env_update(data)


class ImprovedPygameVisualizer:
    """Improved Pygame visualizer with observer pattern support."""
    
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
            'visited': (200, 200, 255),    # Light blue
            'quiet_spot': (0, 255, 0),     # Green for quietest place
            'path': (255, 255, 0)          # Yellow for agent path
        }
        
        # Track visited cells
        self.visited_cells = set()
        self.agent_path = []  # Track agent movement path
        
        # Subject for observer pattern
        self.subject = VisualizationSubject()
        
        # Register observers
        self.agent_observer = AgentPositionObserver(self)
        self.source_observer = SoundSourceObserver(self)
        self.env_observer = EnvironmentStateObserver(self)
        
        self.subject.attach(self.agent_observer)
        self.subject.attach(self.source_observer)
        self.subject.attach(self.env_observer)
    
    def handle_agent_move(self, old_pos, new_pos):
        """Handle agent movement event."""
        self.visited_cells.add(old_pos)
        self.agent_path.append(old_pos)
        # Optionally limit path length to avoid memory issues
        if len(self.agent_path) > 100:
            self.agent_path.pop(0)
    
    def handle_agent_reset(self, position):
        """Handle agent reset event."""
        self.visited_cells.clear()
        self.agent_path.clear()
        self.visited_cells.add(position)
    
    def handle_source_added(self, source):
        """Handle source addition event."""
        pass  # Could trigger animation or effects
    
    def handle_source_moved(self, old_pos, new_pos):
        """Handle source movement event."""
        pass  # Could trigger animation or effects
    
    def handle_source_removed(self, source):
        """Handle source removal event."""
        pass  # Could trigger animation or effects
    
    def handle_env_update(self, data):
        """Handle general environment update."""
        pass  # Could refresh display or trigger effects
    
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
        
        # Draw agent path
        for pos in self.agent_path:
            x, y = pos
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                  self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.colors['path'], rect)
        
        # Draw sound sources
        for source in env.grid_world.sound_sources:
            x, y = source.x, source.y
            center = (y * self.cell_size + self.cell_size // 2, 
                     x * self.cell_size + self.cell_size // 2)
            radius = self.cell_size // 3
            pygame.draw.circle(self.screen, self.colors['sound_source'], center, radius)
        
        # Highlight quietest place if known (for FindQuietestPlaceTask)
        if hasattr(env, 'quietest_cell') and env.quietest_cell:
            x, y = env.quietest_cell
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                  self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.colors['quiet_spot'], rect)
        
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
        # Handle pygame events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise KeyboardInterrupt("Pygame window closed")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise KeyboardInterrupt("Escape key pressed")
        
        # Notify observers of environment update
        self.subject.notify('env_step', {'environment': env})
        
        self.draw_grid(env)
        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS
    
    def close(self):
        """Close the Pygame window and clean up resources."""
        pygame.quit()
        # Detach all observers
        self.subject.detach(self.agent_observer)
        self.subject.detach(self.source_observer)
        self.subject.detach(self.env_observer)


class EventLoggerObserver(Observer):
    """Observer that logs visualization events."""
    
    def __init__(self, log_file='visualization.log'):
        self.log_file = log_file
    
    def update(self, event_type, data):
        """Log visualization events to file."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"[{timestamp}] {event_type}: {data}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)


class PerformanceMonitorObserver(Observer):
    """Observer that monitors visualization performance."""
    
    def __init__(self):
        self.frame_times = []
        self.render_count = 0
    
    def update(self, event_type, data):
        """Monitor performance metrics."""
        if event_type == 'render_complete':
            self.render_count += 1
            # Could track frame times or other performance metrics


def visualize_environment_with_observers(env, title="Sound Navigation Environment"):
    """
    Visualize the environment using Pygame with observer pattern.
    
    Args:
        env: Environment to visualize
        title: Window title
    """
    # Create visualizer with observers
    viz = ImprovedPygameVisualizer()
    pygame.display.set_caption(title)
    
    # Add logging observer
    logger = EventLoggerObserver()
    viz.subject.attach(logger)
    
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
        viz.subject.notify('env_render', {'environment': env})
        viz.draw_grid(env)
        viz.subject.notify('render_complete', {})
        pygame.display.flip()
        clock.tick(60)
    
    # Clean up
    viz.subject.detach(logger)
    viz.close()


def test_visualization_with_observers():
    """
    Test function to demonstrate the visualization with observers.
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
    viz = ImprovedPygameVisualizer()
    
    # Add observers
    logger = EventLoggerObserver()
    perf_monitor = PerformanceMonitorObserver()
    
    viz.subject.attach(logger)
    viz.subject.attach(perf_monitor)
    
    running = True
    step = 0
    
    while running and step < 100:
        try:
            # Update visualization
            viz.update(env)
        except KeyboardInterrupt:
            print("\nVisualization window closed or Escape pressed. Ending test...")
            break

        # Take a random action
        action = np.random.randint(0, 5)
        obs, reward, done = env.step(action)
        
        if done:
            env.reset()
        
        step += 1
    
    # Clean up observers
    viz.subject.detach(logger)
    viz.subject.detach(perf_monitor)
    viz.close()


if __name__ == "__main__":
    test_visualization_with_observers()