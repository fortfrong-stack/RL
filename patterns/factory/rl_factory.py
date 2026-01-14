"""
Factory pattern implementation for RL agents and components.
"""

from abc import ABC, abstractmethod
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some functionality will be disabled.")
    # Define a dummy class for nn.Module when torch is not available
    class nn:
        class Module:
            pass

from rl.dqn import DQNAgent, DQNAgentWrapper


class RLAgentFactory(ABC):
    """Abstract factory for creating RL agents."""
    
    @abstractmethod
    def create_agent(self, input_size, output_size, **kwargs):
        """Create an RL agent."""
        pass


class DQNFactory(RLAgentFactory):
    """Factory for creating DQN agents."""
    
    def create_agent(self, input_size, output_size, **kwargs):
        """Create a DQN agent wrapper."""
        return DQNAgentWrapper(
            input_size=input_size,
            output_size=output_size,
            lr=kwargs.get('lr', 0.001),
            gamma=kwargs.get('gamma', 0.99),
            epsilon=kwargs.get('epsilon', 1.0),
            epsilon_decay=kwargs.get('epsilon_decay', 0.995),
            epsilon_min=kwargs.get('epsilon_min', 0.01),
            target_update_freq=kwargs.get('target_update_freq', 100),
            batch_size=kwargs.get('batch_size', 32),
            buffer_size=kwargs.get('buffer_size', 10000)
        )


class A3CAgent:
    """
    Asynchronous Advantage Actor-Critic (A3C) implementation.
    """
    
    def __init__(self, input_size, output_size, hidden_size=128, learning_rate=0.001):
        """
        Initialize the A3C agent.
        
        Args:
            input_size: Size of the input observation vector
            output_size: Number of possible actions
            hidden_size: Size of hidden layers
            learning_rate: Learning rate for optimization
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use A3CAgent")
            
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Shared network for both policy and value
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_layer = nn.Linear(hidden_size // 2, output_size)
        
        # Value head (critic)
        self.value_layer = nn.Linear(hidden_size // 2, 1)
        
        # Optimizer
        self.optimizer = optim.Adam(list(self.shared_layers.parameters()) + 
                                   list(self.policy_layer.parameters()) + 
                                   list(self.value_layer.parameters()), 
                                   lr=learning_rate)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input observation tensor
            
        Returns:
            Tuple of (policy_logits, value)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use A3CAgent")
            
        shared_output = self.shared_layers(x)
        policy_logits = self.policy_layer(shared_output)
        value = self.value_layer(shared_output)
        
        return policy_logits, value
    
    def get_action(self, state, training=True):
        """
        Get action based on current state.
        
        Args:
            state: Current state observation
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use A3CAgent")
            
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        policy_logits, _ = self.forward(state)
        probs = torch.softmax(policy_logits, dim=-1)
        
        if training:
            action = torch.multinomial(probs, 1).item()
        else:
            action = torch.argmax(probs, dim=-1).item()
        
        return action


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) implementation.
    """
    
    def __init__(self, input_size, output_size, hidden_size=128, learning_rate=0.001, 
                 clip_epsilon=0.2, epochs=4):
        """
        Initialize the PPO agent.
        
        Args:
            input_size: Size of the input observation vector
            output_size: Number of possible actions
            hidden_size: Size of hidden layers
            learning_rate: Learning rate for optimization
            clip_epsilon: Clipping parameter for PPO
            epochs: Number of PPO epochs
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PPOAgent")
            
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        
        # Actor network (policy)
        self.actor_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Critic network (value function)
        self.critic_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=learning_rate)
        
    def get_action(self, state, training=True):
        """
        Get action based on current state.
        
        Args:
            state: Current state observation
            training: Whether in training mode
            
        Returns:
            Selected action and log probability
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PPOAgent")
            
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action probabilities from actor
        action_probs = torch.softmax(self.actor_network(state), dim=-1)
        
        if training:
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[0, action])
        else:
            action = torch.argmax(action_probs, dim=-1).item()
            log_prob = torch.log(action_probs[0, action])
        
        return action, log_prob
    
    def evaluate(self, state):
        """
        Evaluate state value.
        
        Args:
            state: Current state observation
            
        Returns:
            Action probabilities and state value
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PPOAgent")
            
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        action_probs = torch.softmax(self.actor_network(state), dim=-1)
        state_value = self.critic_network(state)
        
        return action_probs, state_value


class A3CFactory(RLAgentFactory):
    """Factory for creating A3C agents."""
    
    def create_agent(self, input_size, output_size, **kwargs):
        """Create an A3C agent."""
        return A3CAgent(
            input_size=input_size,
            output_size=output_size,
            hidden_size=kwargs.get('hidden_size', 128),
            learning_rate=kwargs.get('learning_rate', 0.001)
        )


class PPOFactory(RLAgentFactory):
    """Factory for creating PPO agents."""
    
    def create_agent(self, input_size, output_size, **kwargs):
        """Create a PPO agent."""
        return PPOAgent(
            input_size=input_size,
            output_size=output_size,
            hidden_size=kwargs.get('hidden_size', 128),
            learning_rate=kwargs.get('learning_rate', 0.001),
            clip_epsilon=kwargs.get('clip_epsilon', 0.2),
            epochs=kwargs.get('epochs', 4)
        )


class VisualizationFactory:
    """Factory for creating visualization components."""
    
    @staticmethod
    def create_visualizer(viz_type, **kwargs):
        """
        Create a visualization component based on type.
        
        Args:
            viz_type: Type of visualizer ('pygame', 'matplotlib', etc.)
            **kwargs: Additional arguments for the visualizer
            
        Returns:
            Created visualizer instance
        """
        if viz_type == 'pygame':
            from utils.visualization import PygameVisualizer
            return PygameVisualizer(
                grid_width=kwargs.get('grid_width', 25),
                grid_height=kwargs.get('grid_height', 25),
                cell_size=kwargs.get('cell_size', 20)
            )
        elif viz_type == 'console':
            from interface.console_ui import ConsoleVisualizer
            return ConsoleVisualizer()
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")


class EnvironmentFactory:
    """Factory for creating task environments."""
    
    @staticmethod
    def create_environment(task_type, **kwargs):
        """
        Create a task environment based on type.
        
        Args:
            task_type: Type of task (1, 2, or 3)
            **kwargs: Additional arguments for the environment
            
        Returns:
            Created environment instance
        """
        from core.tasks import create_task_environment
        return create_task_environment(task_type, **kwargs)