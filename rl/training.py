"""
Enhanced training module for DQN agents on sound-based navigation tasks with improved reward normalization.
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Training functionality will be disabled.")

import numpy as np
import sys
import os
import pickle
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    from .dqn import DQNAgentWrapper
    from ..utils.environment_gen import generate_random_environment
    from ..core.tasks import create_task_environment
except ImportError:
    from rl.dqn import DQNAgentWrapper
    from core.tasks import create_task_environment
    from utils.environment_gen import generate_random_environment


def get_observation_size(task_type):
    """
    Get the size of the observation vector for a given task type.
    This is based on the audio features returned by get_audio_observation_features.
    
    Args:
        task_type: Type of task (1, 2, or 3)
    
    Returns:
        Size of the observation vector
    """
    # Import here to avoid circular imports
    from utils.audio_processing import get_audio_observation_features
    sample_obs = get_audio_observation_features(0.5, 0.5)
    return len(sample_obs)


class TrainingStats:
    """
    Class to collect and manage training statistics for enhanced tasks.
    """
    def __init__(self, task_type):
        self.task_type = task_type
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.epsilon_values = []
        self.loss_values = []
        self.reward_stats = []  # Track reward statistics
        self.timestamp = datetime.now().isoformat()
        
    def add_episode_data(self, reward, length, success, epsilon):
        """
        Add data for a completed episode.
        
        Args:
            reward: Total reward for the episode
            length: Number of steps in the episode
            success: Whether the episode was successful
            epsilon: Current epsilon value
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_successes.append(success)
        self.epsilon_values.append(epsilon)
        
    def add_loss(self, loss):
        """
        Add loss value from training step.
        
        Args:
            loss: Loss value from the training step
        """
        self.loss_values.append(loss)
        
    def add_reward_stats(self, avg_reward, min_reward, max_reward):
        """
        Add reward statistics for monitoring.
        
        Args:
            avg_reward: Average reward per step
            min_reward: Minimum reward in episode
            max_reward: Maximum reward in episode
        """
        self.reward_stats.append({
            'avg_reward': avg_reward,
            'min_reward': min_reward,
            'max_reward': max_reward
        })
        
    def get_summary(self):
        """
        Get a summary of the training statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'task_type': self.task_type,
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'success_rate': np.mean(self.episode_successes) if self.episode_successes else 0,
            'timestamp': self.timestamp
        }
        
    def save_to_file(self, filepath):
        """
        Save statistics to a JSON file.
        
        Args:
            filepath: Path to save the statistics
        """
        stats_dict = self.get_summary()
        stats_dict['episode_rewards'] = self.episode_rewards
        stats_dict['episode_lengths'] = self.episode_lengths
        stats_dict['episode_successes'] = self.episode_successes
        stats_dict['epsilon_values'] = self.epsilon_values
        stats_dict['reward_stats'] = self.reward_stats
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(stats_dict, f, indent=2)
            
    def plot_statistics(self, save_path=None):
        """
        Plot training statistics using matplotlib.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'Enhanced Training Statistics - Task {self.task_type}')
            
            # Plot rewards over episodes
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Total Reward per Episode')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            
            # Plot episode lengths
            axes[0, 1].plot(self.episode_lengths)
            axes[0, 1].set_title('Episode Length')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            
            # Plot epsilon decay
            axes[0, 2].plot(self.epsilon_values)
            axes[0, 2].set_title('Epsilon Decay')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Epsilon')
            
            # Plot success rate over time (rolling window)
            if len(self.episode_successes) >= 10:
                rolling_success = []
                for i in range(10, len(self.episode_successes)):
                    window = self.episode_successes[i-10:i]
                    rolling_success.append(sum(window) / len(window))
                
                axes[1, 0].plot(rolling_success)
                axes[1, 0].set_title('Rolling Success Rate (window=10)')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Success Rate')
            else:
                axes[1, 0].text(0.5, 0.5, 'Need more episodes for success rate plot', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Success Rate')
            
            # Plot loss values
            if len(self.loss_values) > 0:
                axes[1, 1].plot(self.loss_values)
                axes[1, 1].set_title('Loss Values')
                axes[1, 1].set_xlabel('Training Step')
                axes[1, 1].set_ylabel('Loss')
            else:
                axes[1, 1].text(0.5, 0.5, 'No loss data available', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Loss Values')
                
            # Plot average reward per step
            if len(self.reward_stats) > 0:
                avg_rewards = [stat['avg_reward'] for stat in self.reward_stats if stat['avg_reward'] is not None]
                if avg_rewards:
                    axes[1, 2].plot(avg_rewards)
                    axes[1, 2].set_title('Avg Reward per Step')
                    axes[1, 2].set_xlabel('Episode')
                    axes[1, 2].set_ylabel('Avg Reward/Step')
            else:
                axes[1, 2].text(0.5, 0.5, 'No reward stats available', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Avg Reward per Step')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
            plt.close()
        except ImportError:
            print("Matplotlib not available, skipping plot generation.")


def train_task(task_type, num_episodes=1000, save_model=True, model_path=None, save_stats=True, stats_path=None):
    """
    Train a DQN agent for a specific task using enhanced reward system.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        num_episodes: Number of episodes to train
        save_model: Whether to save the trained model
        model_path: Path to save the model (default: models/dqn_task_{task_type}.pth)
        save_stats: Whether to save training statistics
        stats_path: Path to save statistics (default: stats/stats_task_{task_type}.json)
    
    Returns:
        Trained DQNAgentWrapper and TrainingStats object
    """
    # Get observation size
    obs_size = get_observation_size(task_type)
    action_size = 5  # up, down, left, right, stay
    
    # Initialize agent with enhanced hyperparameters for better reward handling
    agent = DQNAgentWrapper(
        input_size=obs_size,
        output_size=action_size,
        lr=0.0005 if task_type == 1 else (0.0003 if task_type == 2 else 0.0004),  # Lower learning rates for stability
        gamma=0.95,  # Slightly lower gamma for faster convergence
        epsilon=1.0,
        epsilon_decay=0.995 if task_type == 1 else (0.99 if task_type == 2 else 0.993),  # Adjusted epsilon decays
        epsilon_min=0.01,
        target_update_freq=200,  # Update target network less frequently
        batch_size=64,  # Larger batch size for more stable updates
        buffer_size=50000  # Larger buffer for better experience replay
    )
    
    # Initialize statistics collector
    stats = TrainingStats(task_type)
    
    # Set default paths
    if model_path is None:
        model_path = f"models/dqn_task_{task_type}.pth"
        os.makedirs("models", exist_ok=True)
    
    if stats_path is None:
        stats_path = f"stats/stats_task_{task_type}.json"
        os.makedirs("stats", exist_ok=True)
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc=f"Training Task {task_type}", unit="episode"):
        # Create enhanced environment for this episode
        env = create_task_environment(task_type)
        
        # Get initial observation
        state = env.reset()
        
        total_reward = 0
        step_count = 0
        episode_rewards = []  # Track individual step rewards for statistics
        
        while not env.done and step_count < env.max_steps:
            # Select action using epsilon-greedy
            action = agent.act(state, training=True)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Track reward statistics
            total_reward += reward
            episode_rewards.append(reward)
            
            # Train the agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    stats.add_loss(loss)
            
            # Update state and tracking variables
            state = next_state
            step_count += 1
        
        # Calculate episode reward statistics
        if episode_rewards:
            avg_reward_per_step = np.mean(episode_rewards)
            min_reward = np.min(episode_rewards)
            max_reward = np.max(episode_rewards)
            stats.add_reward_stats(avg_reward_per_step, min_reward, max_reward)
        
        # Determine if the episode was successful based on task type
        success = False
        if hasattr(env, 'found_sources'):  # EnhancedFindAllSourcesTask
            if task_type == 1:  # Find all sources
                success = len(env.found_sources) == len(env.grid_world.sound_sources)
            elif task_type == 2:  # Find quietest place
                if env.quietest_cell:
                    agent_pos = env.agent.get_position()
                    success = agent_pos == env.quietest_cell
            elif task_type == 3:  # Follow moving source
                if env.grid_world.sound_sources:
                    source = env.grid_world.sound_sources[0]
                    agent_x, agent_y = env.agent.get_position()
                    distance = abs(agent_x - source.x) + abs(agent_y - source.y)
                    success = distance < 2  # Close enough to "catch" the source
        
        # Add episode data to statistics
        stats.add_episode_data(total_reward, step_count, success, agent.epsilon)
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f"Task {task_type}, Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Steps: {step_count}")
    
    # Save the model if requested
    if save_model:
        agent.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Save the statistics if requested
    if save_stats:
        stats.save_to_file(stats_path)
        print(f"Statistics saved to {stats_path}")
        
        # Generate plots for the statistics
        plot_path = f"plots/training_task_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        os.makedirs("plots", exist_ok=True)
        stats.plot_statistics(plot_path)
        print(f"Training plots saved to {plot_path}")
    
    return agent, stats


def evaluate_agent(agent, task_type, num_episodes=10, render=False):
    """
    Evaluate a trained agent on an enhanced task.
    
    Args:
        agent: Trained DQNAgentWrapper
        task_type: Type of task (1, 2, or 3)
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment visually
    
    Returns:
        Average total reward across episodes
    """
    total_rewards = []
    
    # Initialize visualization if requested
    viz = None
    if render:
        try:
            from utils.visualization import PygameVisualizer
            viz = PygameVisualizer()
        except Exception as e:
            print(f"Could not initialize visualization: {e}")
            render = False  # Disable rendering if initialization fails
    
    for episode in tqdm(range(num_episodes), desc=f"Evaluation Task {task_type}", unit="episode"):
        # Create enhanced environment for evaluation
        env = create_task_environment(task_type)
        
        # Get initial observation
        state = env.reset()
        
        total_reward = 0
        step_count = 0
        
        while not env.done and step_count < env.max_steps:
            # Update visualization if enabled
            if render and viz:
                try:
                    viz.update(env)
                except KeyboardInterrupt:
                    print("\nVisualization window closed or Escape pressed. Ending evaluation...")
                    if viz:
                        viz.close()
                    return np.mean(total_rewards) if total_rewards else 0
            
            # Select action (no exploration during evaluation)
            action = agent.act(state, training=False)
            
            # Take action in environment
            state, reward, done = env.step(action)
            
            total_reward += reward
            step_count += 1
        
        total_rewards.append(total_reward)
        print(f"Enhanced Evaluation - Episode {episode + 1}, Total Reward: {total_reward:.2f}, Steps: {step_count}")
    
    # Close visualization if it was opened
    if render and viz:
        viz.close()
    
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    return avg_reward


def load_trained_agent(task_type, model_path=None):
    """
    Load a trained agent from a saved model file.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        model_path: Path to the saved model file
    
    Returns:
        Loaded DQNAgentWrapper
    """
    if model_path is None:
        model_path = f"models/dqn_task_{task_type}.pth"
    
    obs_size = get_observation_size(task_type)
    action_size = 5
    
    agent = DQNAgentWrapper(input_size=obs_size, output_size=action_size)
    agent.load(model_path)
    
    return agent


def train_all_tasks(num_episodes=1000):
    """
    Train agents for all three tasks.
    
    Args:
        num_episodes: Number of episodes to train each agent
    """
    print("Starting training for all tasks...")
    
    for task_type in tqdm([1, 2, 3], desc="Overall Training Progress", unit="task"):
        print(f"\nTraining for Task {task_type}...")
        agent, stats = train_task(task_type, num_episodes=num_episodes)
        
        # Evaluate the trained agent
        print(f"\nEvaluating Task {task_type} agent...")
        evaluate_agent(agent, task_type, num_episodes=5)
    
    print("\nTraining completed for all tasks!")