"""
Strategy pattern implementation for different reinforcement learning algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
import random
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')

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

from rl.dqn import ReplayBuffer


class RLScheduler:
    """Event scheduler for observer pattern implementation."""
    
    def __init__(self):
        self.observers = []
    
    def attach(self, observer):
        """Attach an observer."""
        self.observers.append(observer)
    
    def detach(self, observer):
        """Detach an observer."""
        self.observers.remove(observer)
    
    def notify(self, event_type, data):
        """Notify all observers of an event."""
        for observer in self.observers:
            observer.update(event_type, data)


class TrainingObserver(ABC):
    """Abstract observer for training events."""
    
    @abstractmethod
    def update(self, event_type, data):
        """Update the observer with new event data."""
        pass


class RLStrategy(ABC):
    """Abstract strategy for reinforcement learning algorithms."""
    
    def __init__(self, input_size, output_size, scheduler=None):
        self.input_size = input_size
        self.output_size = output_size
        self.scheduler = scheduler or RLScheduler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def select_action(self, state, training=True):
        """Select an action based on the current state."""
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Update the agent based on the experience."""
        pass
    
    @abstractmethod
    def save(self, filepath):
        """Save the trained model to a file."""
        pass
    
    @abstractmethod
    def load(self, filepath):
        """Load a trained model from a file."""
        pass


class DQNStrategy(RLStrategy):
    """DQN (Deep Q-Network) strategy implementation."""
    
    def __init__(self, input_size, output_size, lr=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 target_update_freq=100, batch_size=32, buffer_size=10000, 
                 scheduler=None):
        super().__init__(input_size, output_size, scheduler)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNStrategy")
        
        # Neural networks
        self.q_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        ).to(self.device)
        
        self.target_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Parameters
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
        """Update the target network with weights from the main network."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNStrategy")
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def select_action(self, state, training=True):
        """Select an action based on the current state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNStrategy")
            
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state = state.unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if training and random.random() <= self.epsilon:
            return random.randrange(self.output_size)
        
        # Get Q-values from network
        with torch.no_grad():
            q_values = self.q_network(state)
            return q_values.max(1)[1].item()
    
    def update(self, state, action, reward, next_state, done):
        """Train the network on a batch of experiences from the replay buffer."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNStrategy")
            
        self.remember(state, action, reward, next_state, done)
        
        if len(self.memory) < self.batch_size:
            return None
            
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
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Notify observers of training update
        self.scheduler.notify('training_step', {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'step': self.steps
        })
        
        return loss.item()
    
    def save(self, filepath):
        """Save the trained model to a file."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNStrategy")
            
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load(self, filepath):
        """Load a trained model from a file."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use DQNStrategy")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        
        # Update target network to match loaded network
        self.update_target_network()


class A3CStrategy(RLStrategy):
    """A3C (Asynchronous Advantage Actor-Critic) strategy implementation."""
    
    def __init__(self, input_size, output_size, lr=0.001, gamma=0.99,
                 entropy_coeff=0.01, scheduler=None):
        super().__init__(input_size, output_size, scheduler)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use A3CStrategy")
        
        # Shared network for both policy and value
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_layer = nn.Linear(64, output_size)
        
        # Value head (critic)
        self.value_layer = nn.Linear(64, 1)
        
        # Optimizer
        self.optimizer = optim.Adam(list(self.shared_layers.parameters()) + 
                                   list(self.policy_layer.parameters()) + 
                                   list(self.value_layer.parameters()), 
                                   lr=lr)
        
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.device = self.device
    
    def forward(self, x):
        """Forward pass through the network."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use A3CStrategy")
            
        shared_output = self.shared_layers(x)
        policy_logits = self.policy_layer(shared_output)
        value = self.value_layer(shared_output)
        
        return policy_logits, value
    
    def select_action(self, state, training=True):
        """Select an action based on the current state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use A3CStrategy")
            
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        policy_logits, _ = self.forward(state)
        probs = torch.softmax(policy_logits, dim=-1)
        
        if training:
            action = torch.multinomial(probs, 1).item()
        else:
            action = torch.argmax(probs, dim=-1).item()
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Update the agent based on the experience."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use A3CStrategy")
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device) if next_state is not None else None
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        
        # Get current policy and value
        policy_logits, value = self.forward(state_tensor)
        current_probs = torch.softmax(policy_logits, dim=-1)
        
        # Calculate advantage using TD error
        with torch.no_grad():
            if next_state_tensor is not None:
                _, next_value = self.forward(next_state_tensor)
                target_value = reward_tensor + self.gamma * next_value.squeeze()
            else:
                target_value = reward_tensor
        
        advantage = target_value - value.squeeze()
        
        # Calculate losses
        # Policy loss (with entropy regularization)
        log_probs = torch.log(current_probs + 1e-10)  # Add small epsilon to avoid log(0)
        selected_log_prob = log_probs[0, action]
        policy_loss = -(selected_log_prob * advantage.detach())
        
        # Entropy loss (to encourage exploration)
        entropy = -(current_probs * log_probs).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coeff * entropy
        
        # Value loss
        value_loss = nn.MSELoss()(value.squeeze(), target_value)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(list(self.shared_layers.parameters()) + 
                                      list(self.policy_layer.parameters()) + 
                                      list(self.value_layer.parameters()), max_norm=1.0)
        self.optimizer.step()
        
        # Notify observers of training update
        self.scheduler.notify('training_step', {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item() if 'entropy' in locals() else 0
        })
        
        return total_loss.item()
    
    def save(self, filepath):
        """Save the trained model to a file."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use A3CStrategy")
            
        torch.save({
            'shared_layers_state_dict': self.shared_layers.state_dict(),
            'policy_layer_state_dict': self.policy_layer.state_dict(),
            'value_layer_state_dict': self.value_layer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """Load a trained model from a file."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use A3CStrategy")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.shared_layers.load_state_dict(checkpoint['shared_layers_state_dict'])
        self.policy_layer.load_state_dict(checkpoint['policy_layer_state_dict'])
        self.value_layer.load_state_dict(checkpoint['value_layer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class PPOStrategy(RLStrategy):
    """PPO (Proximal Policy Optimization) strategy implementation."""
    
    def __init__(self, input_size, output_size, lr=0.001, gamma=0.99,
                 clip_epsilon=0.2, epochs=4, entropy_coeff=0.01, scheduler=None):
        super().__init__(input_size, output_size, scheduler)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PPOStrategy")
        
        # Actor network (policy)
        self.actor_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
        # Critic network (value function)
        self.critic_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.entropy_coeff = entropy_coeff
        self.device = self.device
        
        # Storage for rollout data
        self.rollout_buffer = []
    
    def get_action(self, state, training=True):
        """Get action based on current state."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PPOStrategy")
            
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities from actor
        action_probs = torch.softmax(self.actor_network(state), dim=-1)
        
        if training:
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[0, action])
        else:
            action = torch.argmax(action_probs, dim=-1).item()
            log_prob = torch.log(action_probs[0, action])
        
        return action, log_prob
    
    def select_action(self, state, training=True):
        """Select an action based on the current state."""
        action, _ = self.get_action(state, training)
        return action
    
    def evaluate(self, state):
        """Evaluate state value."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PPOStrategy")
            
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        action_probs = torch.softmax(self.actor_network(state), dim=-1)
        state_value = self.critic_network(state)
        
        return action_probs, state_value
    
    def update(self, state, action, reward, next_state, done):
        """Add experience to rollout buffer and perform PPO update when buffer is full."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PPOStrategy")
        
        # Add experience to rollout buffer
        self.rollout_buffer.append((state, action, reward, next_state, done))
        
        # Perform PPO update if buffer is full (or at episode end)
        if done or len(self.rollout_buffer) >= 32:  # Batch size threshold
            return self._ppo_update()
        
        return None
    
    def _ppo_update(self):
        """Perform PPO update using collected rollout data."""
        if len(self.rollout_buffer) == 0:
            return None
        
        # Process the rollout buffer to compute advantages
        states, actions, rewards, next_states, dones = zip(*self.rollout_buffer)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        
        # Compute state values for all states
        with torch.no_grad():
            state_values = self.critic_network(states_tensor).squeeze()
            
            # Compute advantages using Generalized Advantage Estimation (GAE)
            advantages = torch.zeros_like(rewards_tensor).to(self.device)
            gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0 if dones[t] else self.critic_network(
                        torch.FloatTensor(next_states[t]).unsqueeze(0).to(self.device)
                    ).item()
                else:
                    next_value = state_values[t + 1].item()
                
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - state_values[t].item()
                gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae  # Lambda = 0.95
                advantages[t] = gae
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store old action probabilities for ratio calculation
        old_action_probs = torch.softmax(self.actor_network(states_tensor), dim=-1)
        old_log_probs = torch.log(old_action_probs + 1e-10)
        old_selected_log_probs = old_log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
        
        # Perform multiple PPO epochs
        for _ in range(self.epochs):
            # Get current policy and value estimates
            curr_action_probs = torch.softmax(self.actor_network(states_tensor), dim=-1)
            curr_log_probs = torch.log(curr_action_probs + 1e-10)
            curr_selected_log_probs = curr_log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
            
            # Calculate ratio
            ratio = torch.exp(curr_selected_log_probs - old_selected_log_probs)
            
            # Calculate surrogate objectives
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            state_values_new = self.critic_network(states_tensor).squeeze()
            value_loss = nn.MSELoss()(state_values_new, rewards_tensor + advantages)  # Using rewards + advantages as targets
            
            # Calculate entropy for exploration
            entropy = -(curr_action_probs * torch.log(curr_action_probs + 1e-10)).sum(dim=-1).mean()
            
            # Total losses
            total_actor_loss = actor_loss - self.entropy_coeff * entropy
            total_critic_loss = value_loss
            
            # Update actor
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            total_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=1.0)
            self.critic_optimizer.step()
        
        # Clear the rollout buffer
        self.rollout_buffer.clear()
        
        # Notify observers of training update
        self.scheduler.notify('training_step', {
            'actor_loss': actor_loss.item(),
            'critic_loss': value_loss.item(),
            'entropy': entropy.item(),
            'buffer_size': len(self.rollout_buffer)
        })
        
        return total_actor_loss.item() + total_critic_loss.item()
    
    def save(self, filepath):
        """Save the trained model to a file."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PPOStrategy")
            
        torch.save({
            'actor_network_state_dict': self.actor_network.state_dict(),
            'critic_network_state_dict': self.critic_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """Load a trained model from a file."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PPOStrategy")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_network.load_state_dict(checkpoint['actor_network_state_dict'])
        self.critic_network.load_state_dict(checkpoint['critic_network_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class TrainingMetricsObserver(TrainingObserver):
    """Observer that collects training metrics."""
    
    def __init__(self):
        self.metrics = {
            'losses': [],
            'entropies': [],
            'steps': 0
        }
    
    def update(self, event_type, data):
        """Update metrics based on training events."""
        if event_type == 'training_step':
            if 'loss' in data:
                self.metrics['losses'].append(data['loss'])
            if 'entropy' in data:
                self.metrics['entropies'].append(data['entropy'])
            self.metrics['steps'] += 1


class EventLoggingObserver(TrainingObserver):
    """Observer that logs training events."""
    
    def __init__(self, log_file='training.log'):
        self.log_file = log_file
    
    def update(self, event_type, data):
        """Log training events to file."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"[{timestamp}] {event_type}: {data}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)