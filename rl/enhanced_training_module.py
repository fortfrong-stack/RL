"""
Enhanced training module with advanced features:
- Detailed metrics and logging
- Checkpoint creation and loading
- Improved progress tracking
"""

import numpy as np
import os
import json
import pickle
from datetime import datetime
from tqdm import tqdm
import torch

from .dqn import DQNAgentWrapper
from .training import TrainingStats, get_observation_size
from core.tasks import create_task_environment


class EnhancedTrainingStats(TrainingStats):
    """
    Enhanced version of TrainingStats with additional metrics and logging capabilities.
    """
    
    def __init__(self, task_type):
        super().__init__(task_type)
        self.action_distribution = []  # Track distribution of actions taken
        self.exploration_rate = []     # Track exploration vs exploitation
        self.q_values_history = []     # Track Q-values over time
        self.gradient_norms = []       # Track gradient norms for debugging
        self.learning_rate_history = []  # Track learning rate changes
        self.episode_success_rates = []  # Track success rate over windows
        self.step_rewards = []         # Track rewards per step within episodes
        self.value_consistency = []    # Track consistency of value estimates
    
    def add_action_distribution(self, action_counts):
        """Add action distribution for the episode."""
        self.action_distribution.append(action_counts)
    
    def add_exploration_metric(self, exploratory_actions, total_actions):
        """Add exploration metric."""
        self.exploration_rate.append(exploratory_actions / max(total_actions, 1))
    
    def add_q_values(self, q_values):
        """Add Q-values from an episode."""
        self.q_values_history.append(q_values)
    
    def add_gradient_norm(self, grad_norm):
        """Add gradient norm for debugging."""
        self.gradient_norms.append(grad_norm)
    
    def add_learning_rate(self, lr):
        """Add learning rate."""
        self.learning_rate_history.append(lr)
    
    def add_episode_success_rate(self, recent_successes, window_size):
        """Add recent success rate."""
        self.episode_success_rates.append(recent_successes / window_size)
    
    def add_step_rewards(self, rewards):
        """Add step-by-step rewards for the episode."""
        self.step_rewards.append(rewards)
    
    def get_detailed_summary(self):
        """Get a detailed summary of training statistics."""
        summary = self.get_summary()
        
        # Additional metrics
        if self.action_distribution:
            avg_action_dist = np.mean(self.action_distribution, axis=0)
            summary['avg_action_distribution'] = avg_action_dist.tolist()
        
        if self.exploration_rate:
            summary['avg_exploration_rate'] = np.mean(self.exploration_rate)
        
        if self.gradient_norms:
            summary['gradient_norm_stats'] = {
                'mean': float(np.mean(self.gradient_norms)),
                'std': float(np.std(self.gradient_norms)),
                'min': float(np.min(self.gradient_norms)),
                'max': float(np.max(self.gradient_norms))
            }
        
        if self.learning_rate_history:
            summary['learning_rate_range'] = {
                'min': min(self.learning_rate_history),
                'max': max(self.learning_rate_history)
            }
        
        if self.episode_success_rates:
            summary['recent_success_rate'] = float(np.mean(self.episode_success_rates[-10:]))
        
        return summary


class CheckpointManager:
    """
    Manager for creating, saving, and loading training checkpoints.
    """
    
    def __init__(self, checkpoint_dir="checkpoints", keep_best_n=5):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_best_n = keep_best_n
        self.best_checkpoints = []  # List of (score, filepath) tuples
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, agent, episode, stats=None, score=None, additional_data=None):
        """
        Save a training checkpoint.
        
        Args:
            agent: DQNAgentWrapper to save
            episode: Current episode number
            stats: TrainingStats object
            score: Score to rank this checkpoint (higher is better)
            additional_data: Any additional data to save with checkpoint
        """
        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_task_{getattr(agent, '_task_type', 'unknown')}_ep_{episode}_{timestamp}.pth"
        )
        
        # Prepare checkpoint data
        checkpoint_data = {
            'episode': episode,
            'agent_state': {
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'steps': agent.steps
            },
            'timestamp': datetime.now().isoformat(),
            'additional_data': additional_data or {}
        }
        
        # Add stats if provided
        if stats is not None:
            checkpoint_data['stats'] = stats.get_detailed_summary()
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Track best checkpoints if score is provided
        if score is not None:
            self._track_best_checkpoint(score, checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, agent, checkpoint_path, load_stats=False):
        """
        Load a training checkpoint.
        
        Args:
            agent: DQNAgentWrapper to load into
            checkpoint_path: Path to checkpoint file
            load_stats: Whether to load stats as well
        
        Returns:
            Loaded agent, episode number, and optionally stats
        """
        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        
        # Load agent state
        agent.q_network.load_state_dict(checkpoint['agent_state']['q_network_state_dict'])
        agent.target_network.load_state_dict(checkpoint['agent_state']['target_network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['agent_state']['optimizer_state_dict'])
        agent.epsilon = checkpoint['agent_state']['epsilon']
        agent.steps = checkpoint['agent_state']['steps']
        
        # Update target network to match loaded network
        agent.update_target_network()
        
        episode = checkpoint['episode']
        timestamp = checkpoint.get('timestamp', 'unknown')
        
        print(f"Loaded checkpoint from {checkpoint_path} (episode {episode}, timestamp {timestamp})")
        
        if load_stats and 'stats' in checkpoint:
            return agent, episode, checkpoint['stats']
        else:
            return agent, episode
    
    def _track_best_checkpoint(self, score, filepath):
        """Track the best checkpoints to potentially remove old ones."""
        self.best_checkpoints.append((score, filepath))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)  # Higher scores are better
        
        # Remove extra checkpoints if we have too many
        if len(self.best_checkpoints) > self.keep_best_n:
            # Remove the lowest-scoring checkpoint
            _, oldest_path = self.best_checkpoints.pop()
            if os.path.exists(oldest_path):
                os.remove(oldest_path)
                print(f"Removed old checkpoint: {oldest_path}")
    
    def list_checkpoints(self):
        """List all available checkpoints."""
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(self.checkpoint_dir, filename)
                checkpoints.append(filepath)
        return sorted(checkpoints)
    
    def get_best_checkpoint(self):
        """Get the path to the best checkpoint based on score."""
        if self.best_checkpoints:
            return self.best_checkpoints[0][1]  # Return path of highest scoring checkpoint
        else:
            # If no tracked best, return the most recent checkpoint
            all_checkpoints = self.list_checkpoints()
            return all_checkpoints[-1] if all_checkpoints else None


def train_with_enhanced_monitoring(
    task_type, 
    num_episodes=1000, 
    save_checkpoints=True, 
    checkpoint_freq=100, 
    save_model=True, 
    model_path=None,
    log_detailed_metrics=True
):
    """
    Train an agent with enhanced monitoring, checkpoints, and detailed metrics.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        num_episodes: Number of episodes to train
        save_checkpoints: Whether to save checkpoints
        checkpoint_freq: How often to save checkpoints
        save_model: Whether to save the final model
        model_path: Path to save the final model
        log_detailed_metrics: Whether to log detailed metrics
    
    Returns:
        Trained agent and EnhancedTrainingStats
    """
    # Get observation size
    obs_size = get_observation_size(task_type)
    action_size = 5  # up, down, left, right, stay
    
    # Initialize agent with enhanced hyperparameters
    agent = DQNAgentWrapper(
        input_size=obs_size,
        output_size=action_size,
        lr=0.0005 if task_type == 1 else (0.0003 if task_type == 2 else 0.0004),
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995 if task_type == 1 else (0.99 if task_type == 2 else 0.993),
        epsilon_min=0.01,
        target_update_freq=200,
        batch_size=64,
        buffer_size=50000
    )
    
    # Set task type for checkpoint naming
    agent._task_type = task_type
    
    # Initialize enhanced statistics
    stats = EnhancedTrainingStats(task_type)
    
    # Initialize checkpoint manager
    checkpoint_manager = None
    if save_checkpoints:
        checkpoint_manager = CheckpointManager()
    
    # Set default paths
    if model_path is None:
        model_path = f"models/enhanced_dqn_task_{task_type}.pth"
        os.makedirs("models", exist_ok=True)
    
    # Training loop with enhanced monitoring
    for episode in tqdm(range(num_episodes), desc=f"Enhanced Training Task {task_type}", unit="episode"):
        # Create environment for this episode
        env = create_task_environment(task_type)
        state = env.reset()
        
        total_reward = 0
        step_count = 0
        episode_rewards = []
        episode_actions = []
        exploratory_actions = 0  # Count of exploratory actions taken
        
        # Initialize action counter
        action_counts = [0, 0, 0, 0, 0]  # For 5 possible actions
        
        while not env.done and step_count < env.max_steps:
            # Select action using epsilon-greedy
            action = agent.act(state, training=True)
            
            # Track if this was an exploratory action
            if np.random.random() < agent.epsilon:
                exploratory_actions += 1
            
            # Track action taken
            action_counts[action] += 1
            episode_actions.append(action)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Track reward statistics
            total_reward += reward
            episode_rewards.append(reward)
            
            # Train the agent if buffer is large enough
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    stats.add_loss(loss)
            
            # Update state and tracking variables
            state = next_state
            step_count += 1
        
        # Calculate and store episode statistics
        if episode_rewards:
            avg_reward_per_step = np.mean(episode_rewards)
            min_reward = np.min(episode_rewards)
            max_reward = np.max(episode_rewards)
            stats.add_reward_stats(avg_reward_per_step, min_reward, max_reward)
        
        # Determine if the episode was successful
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
        
        # Add detailed metrics if requested
        if log_detailed_metrics:
            stats.add_action_distribution(action_counts)
            stats.add_exploration_metric(exploratory_actions, step_count)
            stats.add_learning_rate(agent.optimizer.param_groups[0]['lr'])
        
        # Save checkpoint if needed
        if save_checkpoints and (episode + 1) % checkpoint_freq == 0:
            checkpoint_manager.save_checkpoint(
                agent=agent,
                episode=episode + 1,
                stats=stats,
                score=total_reward,  # Use total reward as the score for checkpoint ranking
                additional_data={
                    'episode': episode + 1,
                    'avg_reward': total_reward,
                    'success': success
                }
            )
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f"Task {task_type}, Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Steps: {step_count}, Success: {success}")
    
    # Save final checkpoint
    if save_checkpoints:
        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            agent=agent,
            episode=num_episodes,
            stats=stats,
            score=np.mean(stats.episode_rewards) if stats.episode_rewards else 0,
            additional_data={'final_model': True}
        )
        print(f"Final checkpoint saved: {final_checkpoint_path}")
    
    # Save the final model if requested
    if save_model:
        agent.save(model_path)
        print(f"Final model saved to {model_path}")
    
    return agent, stats


def resume_training_from_checkpoint(
    task_type,
    checkpoint_path,
    additional_episodes=500,
    save_model=True,
    model_path=None
):
    """
    Resume training from a saved checkpoint.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        checkpoint_path: Path to the checkpoint to resume from
        additional_episodes: Number of additional episodes to train
        save_model: Whether to save the resumed model
        model_path: Path to save the resumed model
    
    Returns:
        Resumed agent and updated stats
    """
    # Get observation size
    obs_size = get_observation_size(task_type)
    action_size = 5
    
    # Initialize agent
    agent = DQNAgentWrapper(
        input_size=obs_size,
        output_size=action_size
    )
    
    # Load from checkpoint
    checkpoint_manager = CheckpointManager()
    agent, start_episode = checkpoint_manager.load_checkpoint(agent, checkpoint_path)
    
    # Continue training
    print(f"Resuming training from episode {start_episode} for {additional_episodes} more episodes...")
    
    # Initialize stats (we won't have previous stats, so start fresh)
    stats = EnhancedTrainingStats(task_type)
    
    # Set default model path if not provided
    if model_path is None:
        model_path = f"models/resumed_dqn_task_{task_type}.pth"
        os.makedirs("models", exist_ok=True)
    
    # Training loop
    for episode_idx in tqdm(range(additional_episodes), desc=f"Resumed Training Task {task_type}", unit="episode"):
        episode_num = start_episode + episode_idx
        
        # Create environment for this episode
        env = create_task_environment(task_type)
        state = env.reset()
        
        total_reward = 0
        step_count = 0
        episode_rewards = []
        
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
            
            # Train the agent if buffer is large enough
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    stats.add_loss(loss)
            
            # Update state and tracking variables
            state = next_state
            step_count += 1
        
        # Calculate and store episode statistics
        if episode_rewards:
            avg_reward_per_step = np.mean(episode_rewards)
            min_reward = np.min(episode_rewards)
            max_reward = np.max(episode_rewards)
            stats.add_reward_stats(avg_reward_per_step, min_reward, max_reward)
        
        # Determine if the episode was successful
        success = False
        if hasattr(env, 'found_sources'):
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
                    success = distance < 2
        
        # Add episode data to statistics
        stats.add_episode_data(total_reward, step_count, success, agent.epsilon)
        
        # Print progress every 100 episodes
        if episode_idx % 100 == 0:
            print(f"Task {task_type}, Episode {episode_num}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Steps: {step_count}, Success: {success}")
    
    # Save the resumed model if requested
    if save_model:
        agent.save(model_path)
        print(f"Resumed model saved to {model_path}")
    
    return agent, stats


def compare_training_runs(stats_list, names_list):
    """
    Compare multiple training runs side by side.
    
    Args:
        stats_list: List of TrainingStats objects
        names_list: List of names for each run
    
    Returns:
        Comparison dictionary
    """
    comparison = {
        'runs': [],
        'summary_comparison': {}
    }
    
    for i, (stats, name) in enumerate(zip(stats_list, names_list)):
        run_summary = {
            'name': name,
            'avg_reward': float(np.mean(stats.episode_rewards)) if stats.episode_rewards else 0,
            'std_reward': float(np.std(stats.episode_rewards)) if stats.episode_rewards else 0,
            'avg_length': float(np.mean(stats.episode_lengths)) if stats.episode_lengths else 0,
            'success_rate': float(np.mean(stats.episode_successes)) if stats.episode_successes else 0,
            'final_epsilon': stats.epsilon_values[-1] if stats.epsilon_values else 1.0,
            'total_episodes': len(stats.episode_rewards)
        }
        comparison['runs'].append(run_summary)
    
    # Overall comparison
    avg_rewards = [run['avg_reward'] for run in comparison['runs']]
    best_run_idx = np.argmax(avg_rewards)
    comparison['summary_comparison'] = {
        'best_run': comparison['runs'][best_run_idx]['name'],
        'best_avg_reward': float(np.max(avg_rewards)),
        'avg_of_avgs': float(np.mean(avg_rewards))
    }
    
    return comparison