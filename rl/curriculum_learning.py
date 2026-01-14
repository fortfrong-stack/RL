"""
Curriculum learning implementation for gradually increasing difficulty during training.
"""

import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm

from .training import train_task, evaluate_agent, TrainingStats
from .dqn import DQNAgentWrapper


class CurriculumScheduler:
    """
    Scheduler for curriculum learning that gradually increases task difficulty.
    """
    
    def __init__(self, task_type, initial_params=None, difficulty_levels=None):
        """
        Initialize the curriculum scheduler.
        
        Args:
            task_type: Type of task (1, 2, or 3)
            initial_params: Initial parameters for the task
            difficulty_levels: List of difficulty configurations
        """
        self.task_type = task_type
        self.current_level = 0
        self.completed_episodes = 0
        
        # Default difficulty progression for each task type
        if difficulty_levels is None:
            self.difficulty_levels = self._get_default_difficulty_levels(task_type)
        else:
            self.difficulty_levels = difficulty_levels
            
        # Performance thresholds to advance to next level
        self.performance_thresholds = [0.6, 0.7, 0.8]  # Success rate thresholds
        
        # Episode counts for each level
        self.level_episodes = [500, 500, 1000]  # Episodes per level
        
        # Current task parameters
        self.current_params = initial_params or {}
        
    def _get_default_difficulty_levels(self, task_type):
        """Get default difficulty levels for each task type."""
        if task_type == 1:  # Find all sources
            return [
                {'num_sources': 2, 'grid_size': (10, 10)},      # Level 0: Easy
                {'num_sources': 3, 'grid_size': (15, 15)},      # Level 1: Medium  
                {'num_sources': 5, 'grid_size': (20, 20)},      # Level 2: Hard
                {'num_sources': 6, 'grid_size': (25, 25)}       # Level 3: Expert
            ]
        elif task_type == 2:  # Find quietest place
            return [
                {'num_sources': 2, 'grid_size': (10, 10)},      # Level 0: Easy
                {'num_sources': 3, 'grid_size': (15, 15)},      # Level 1: Medium
                {'num_sources': 5, 'grid_size': (20, 20)},      # Level 2: Hard
                {'num_sources': 6, 'grid_size': (25, 25)}       # Level 3: Expert
            ]
        elif task_type == 3:  # Follow moving source
            return [
                {'num_sources': 1, 'grid_size': (10, 10), 'move_interval': 20},  # Level 0: Easy
                {'num_sources': 1, 'grid_size': (15, 15), 'move_interval': 15},  # Level 1: Medium
                {'num_sources': 1, 'grid_size': (20, 20), 'move_interval': 10},  # Level 2: Hard
                {'num_sources': 1, 'grid_size': (25, 25), 'move_interval': 8}    # Level 3: Expert
            ]
        else:
            return [{'num_sources': 3, 'grid_size': (15, 15)}]  # Default
    
    def get_current_params(self):
        """Get current task parameters."""
        return {**self.current_params, **self.difficulty_levels[self.current_level]}
    
    def should_advance_level(self, recent_performance):
        """
        Determine if we should advance to the next difficulty level.
        
        Args:
            recent_performance: Recent performance metric (e.g., success rate)
            
        Returns:
            bool: Whether to advance to next level
        """
        if self.current_level >= len(self.difficulty_levels) - 1:
            return False  # Already at max level
            
        threshold = self.performance_thresholds[min(self.current_level, len(self.performance_thresholds)-1)]
        return recent_performance >= threshold
    
    def advance_level(self):
        """Advance to the next difficulty level."""
        if self.current_level < len(self.difficulty_levels) - 1:
            self.current_level += 1
            self.completed_episodes = 0
            print(f"Advancing to difficulty level {self.current_level}: {self.difficulty_levels[self.current_level]}")
            return True
        return False
    
    def get_remaining_episodes(self):
        """Get remaining episodes for current level."""
        max_episodes = self.level_episodes[min(self.current_level, len(self.level_episodes)-1)]
        return max(0, max_episodes - self.completed_episodes)
    
    def increment_episode(self):
        """Increment episode counter."""
        self.completed_episodes += 1


def train_with_curriculum(task_type, total_episodes=2000, save_model=True, model_path=None):
    """
    Train an agent using curriculum learning approach.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        total_episodes: Total number of episodes to train
        save_model: Whether to save the final model
        model_path: Path to save the model
    
    Returns:
        Trained agent and combined training statistics
    """
    scheduler = CurriculumScheduler(task_type)
    stats_list = []
    cumulative_episodes = 0
    
    print(f"Starting curriculum learning for Task {task_type}")
    
    # Train on each difficulty level
    while cumulative_episodes < total_episodes and scheduler.current_level < len(scheduler.difficulty_levels):
        remaining_episodes = scheduler.get_remaining_episodes()
        if remaining_episodes <= 0:
            # Try to advance to next level
            if not scheduler.advance_level():
                break  # Can't advance further
            continue
        
        # Determine number of episodes for this iteration
        episodes_this_round = min(remaining_episodes, total_episodes - cumulative_episodes)
        
        print(f"Training on difficulty level {scheduler.current_level} for {episodes_this_round} episodes...")
        
        # Train on current difficulty level
        params = scheduler.get_current_params()
        print(f"Current parameters: {params}")
        
        # For now, we'll use the standard training function with modified parameters
        # In a real implementation, you'd need to modify create_task_environment to accept these params
        agent, level_stats = train_task(
            task_type=task_type,
            num_episodes=episodes_this_round,
            save_model=False  # Don't save intermediate models
        )
        
        stats_list.append(level_stats)
        
        # Update scheduler
        for _ in range(episodes_this_round):
            scheduler.increment_episode()
        
        # Calculate recent performance (success rate)
        if level_stats.episode_successes:
            recent_success_rate = np.mean(level_stats.episode_successes[-20:]) if len(level_stats.episode_successes) >= 20 else np.mean(level_stats.episode_successes)
            if scheduler.should_advance_level(recent_success_rate):
                print(f"Performance threshold met ({recent_success_rate:.2f}), advancing level...")
                scheduler.advance_level()
        
        cumulative_episodes += episodes_this_round
    
    # Combine all statistics
    combined_stats = combine_training_stats(stats_list, task_type)
    
    # Save final model if requested
    if save_model:
        if model_path is None:
            model_path = f"models/curriculum_dqn_task_{task_type}.pth"
            os.makedirs("models", exist_ok=True)
        agent.save(model_path)
        print(f"Final curriculum-trained model saved to {model_path}")
    
    return agent, combined_stats


def combine_training_stats(stats_list, task_type):
    """
    Combine multiple training statistics objects into one.
    
    Args:
        stats_list: List of TrainingStats objects
        task_type: Type of task
    
    Returns:
        Combined TrainingStats object
    """
    combined = TrainingStats(task_type)
    
    for stats in stats_list:
        combined.episode_rewards.extend(stats.episode_rewards)
        combined.episode_lengths.extend(stats.episode_lengths)
        combined.episode_successes.extend(stats.episode_successes)
        combined.epsilon_values.extend(stats.epsilon_values)
        combined.loss_values.extend(stats.loss_values)
        combined.reward_stats.extend(stats.reward_stats)
    
    return combined


class TransferLearningManager:
    """
    Manager for implementing transfer learning between tasks.
    """
    
    def __init__(self):
        self.pretrained_models = {}
    
    def save_pretrained_features(self, agent, task_type, feature_path=None):
        """
        Save pretrained features from a trained agent for transfer.
        
        Args:
            agent: Trained DQNAgentWrapper
            task_type: Type of source task
            feature_path: Path to save features
        """
        if feature_path is None:
            feature_path = f"models/pretrained_features_task_{task_type}.pth"
            os.makedirs("models", exist_ok=True)
        
        # Extract the feature extraction layers (all but the last layer)
        feature_dict = {
            'fc1_weight': agent.q_network.fc1.weight.data.clone(),
            'fc1_bias': agent.q_network.fc1.bias.data.clone(),
            'fc2_weight': agent.q_network.fc2.weight.data.clone(),
            'fc2_bias': agent.q_network.fc2.bias.data.clone(),
            'task_type': task_type,
            'timestamp': datetime.now().isoformat()
        }
        
        import torch
        torch.save(feature_dict, feature_path)
        print(f"Pretrained features saved to {feature_path}")
    
    def load_and_transfer(self, target_task_type, source_task_type, feature_path=None):
        """
        Load pretrained features and adapt to a new task.
        
        Args:
            target_task_type: Type of target task
            source_task_type: Type of source task
            feature_path: Path to pretrained features
        
        Returns:
            New agent with transferred features
        """
        from .dqn import get_observation_size
        
        if feature_path is None:
            feature_path = f"models/pretrained_features_task_{source_task_type}.pth"
        
        import torch
        
        # Load pretrained features
        try:
            features = torch.load(feature_path)
        except FileNotFoundError:
            print(f"No pretrained features found at {feature_path}, creating new agent")
            obs_size = get_observation_size(target_task_type)
            return DQNAgentWrapper(input_size=obs_size, output_size=5)
        
        # Create new agent for target task
        obs_size = get_observation_size(target_task_type)
        target_agent = DQNAgentWrapper(input_size=obs_size, output_size=5)
        
        # Transfer the feature extraction layers
        target_agent.q_network.fc1.weight.data.copy_(features['fc1_weight'])
        target_agent.q_network.fc1.bias.data.copy_(features['fc1_bias'])
        target_agent.q_network.fc2.weight.data.copy_(features['fc2_weight'])
        target_agent.q_network.fc2.bias.data.copy_(features['fc2_bias'])
        
        # Keep the last layer random for adaptation to new task
        # This allows the agent to learn task-specific outputs
        
        # Also transfer to target network
        target_agent.target_network.fc1.weight.data.copy_(features['fc1_weight'])
        target_agent.target_network.fc1.bias.data.copy_(features['fc1_bias'])
        target_agent.target_network.fc2.weight.data.copy_(features['fc2_weight'])
        target_agent.target_network.fc2.bias.data.copy_(features['fc2_bias'])
        
        print(f"Transferred features from Task {source_task_type} to Task {target_task_type}")
        return target_agent
    
    def multi_task_training(self, task_sequence, episodes_per_task=500):
        """
        Train on a sequence of tasks with transfer learning between them.
        
        Args:
            task_sequence: List of task types to train on in sequence
            episodes_per_task: Number of episodes per task
        
        Returns:
            Dict mapping task type to trained agent
        """
        agents = {}
        transfer_manager = self
        
        for i, task_type in enumerate(task_sequence):
            print(f"\nTraining on Task {task_type}...")
            
            if i == 0:
                # First task: train from scratch
                agent, stats = train_task(task_type, num_episodes=episodes_per_task)
            else:
                # Subsequent tasks: transfer from previous task
                prev_task_type = task_sequence[i-1]
                agent = transfer_manager.load_and_transfer(task_type, prev_task_type)
                
                # Continue training
                from .training import get_observation_size
                obs_size = get_observation_size(task_type)
                
                # Modify hyperparameters for fine-tuning
                agent.epsilon = 0.3  # Start with higher exploration for new task
                agent.epsilon_min = 0.05
                
                # Training loop adapted for transfer learning
                from core.tasks import create_task_environment
                from utils.audio_processing import get_audio_observation_features
                
                for episode in tqdm(range(episodes_per_task), desc=f"Transfer Training Task {task_type}"):
                    env = create_task_environment(task_type)
                    state = env.reset()
                    
                    total_reward = 0
                    step_count = 0
                    
                    while not env.done and step_count < env.max_steps:
                        action = agent.act(state, training=True)
                        next_state, reward, done = env.step(action)
                        
                        agent.remember(state, action, reward, next_state, done)
                        
                        if len(agent.memory) > agent.batch_size:
                            agent.replay()
                        
                        state = next_state
                        step_count += 1
                        total_reward += reward
                    
                    # Add to statistics (for simplicity, we're not collecting detailed stats here)
                    if episode % 100 == 0:
                        print(f"Task {task_type}, Episode {episode}, Total Reward: {total_reward:.2f}")
            
            agents[task_type] = agent
            # Save pretrained features for potential future transfers
            self.save_pretrained_features(agent, task_type)
        
        return agents


def create_checkpoint_callback(agent, checkpoint_dir="checkpoints", checkpoint_freq=100):
    """
    Create a callback function for saving checkpoints during training.
    
    Args:
        agent: DQNAgentWrapper to save
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Frequency of checkpoints (in episodes)
    
    Returns:
        Function that can be called to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(episode, additional_info=None):
        """Save a checkpoint of the agent."""
        checkpoint_path = f"{checkpoint_dir}/checkpoint_task_{getattr(agent, '_task_type', 'unknown')}_{episode}.pth"
        
        import torch
        checkpoint_data = {
            'q_network_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'steps': agent.steps,
            'episode': episode,
            'additional_info': additional_info or {}
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved at episode {episode}: {checkpoint_path}")
    
    return save_checkpoint


def load_from_checkpoint(agent, checkpoint_path):
    """
    Load an agent from a checkpoint.
    
    Args:
        agent: DQNAgentWrapper to load into
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Updated agent and episode number
    """
    import torch
    
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint['epsilon']
    agent.steps = checkpoint['steps']
    
    # Update target network to match loaded network
    agent.update_target_network()
    
    episode_num = checkpoint.get('episode', 0)
    print(f"Loaded checkpoint from {checkpoint_path}, resuming from episode {episode_num}")
    
    return agent, episode_num