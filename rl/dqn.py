"""
DQN (Deep Q-Network) implementation for the sound-based navigation tasks.
"""

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

import numpy as np
import random
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    from utils.audio_processing import get_audio_observation_features
except ImportError:
    from utils.audio_processing import get_audio_observation_features


class DQNAgent(nn.Module):
    """
    Deep Q-Network agent for sound-based navigation tasks.
    """
    
    def __init__(self, input_size, output_size, hidden_size=128):
        """
        Initialize the DQN agent.
        
        Args:
            input_size: Size of the input observation vector
            output_size: Number of possible actions (5: up, down, left, right, stay)
            hidden_size: Size of hidden layers
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNAgent")
        
        super(DQNAgent, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize with small weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input observation tensor
        
        Returns:
            Q-values for each action
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNAgent")
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    """
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Size of the batch to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """
        Get current size of the buffer.
        
        Returns:
            Current size of the buffer
        """
        return len(self.buffer)


class DQNAgentWrapper:
    """
    Wrapper class that manages the DQN agent and training process.
    """
    
    def __init__(self, input_size, output_size, lr=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 target_update_freq=100, batch_size=32, buffer_size=10000):
        """
        Initialize the DQN agent wrapper.
        
        Args:
            input_size: Size of input observation vector
            output_size: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum exploration rate
            target_update_freq: How often to update target network
            batch_size: Size of training batches
            buffer_size: Size of replay buffer
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNAgentWrapper")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNAgent(input_size, output_size).to(self.device)
        self.target_network = DQNAgent(input_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Training parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Update target network to match main network
        self.update_target_network()
        
        # Training step counter
        self.steps = 0
    
    def update_target_network(self):
        """
        Update the target network with weights from the main network.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNAgentWrapper")
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        """
        Select an action based on the current state.
        
        Args:
            state: Current state observation
            training: Whether in training mode (affects epsilon-greedy)
        
        Returns:
            Selected action
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNAgentWrapper")
            
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state = state.unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if training and random.random() <= self.epsilon:
            return random.randrange(5)  # Random action (0-4)
        
        # Get Q-values from network
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.max(1)[1].item()  # Return action with highest Q-value
    
    def replay(self):
        """
        Train the network on a batch of experiences from the replay buffer.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNAgentWrapper")
            
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
    
    def save(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNAgentWrapper")
            
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNAgentWrapper")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        
        # Update target network to match loaded network
        self.update_target_network()