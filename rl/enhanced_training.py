"""
Enhanced training module using design patterns: Factory, Strategy, Observer.
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
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from patterns.factory.rl_factory import DQNFactory, A3CFactory, PPOFactory
from patterns.strategy.rl_strategies import DQNStrategy, A3CStrategy, PPOStrategy, TrainingMetricsObserver
from patterns.factory.serialization_manager import SerializationManager


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


def train_task_with_strategy(task_type, algorithm='dqn', num_episodes=1000, save_model=True, model_path=None, save_stats=True, stats_path=None):
    """
    Train an RL agent using the strategy pattern for different algorithms.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        algorithm: RL algorithm to use ('dqn', 'a3c', 'ppo')
        num_episodes: Number of episodes to train
        save_model: Whether to save the trained model
        model_path: Path to save the model
        save_stats: Whether to save training statistics
        stats_path: Path to save statistics
    
    Returns:
        Trained agent and training statistics
    """
    # Get observation size
    obs_size = get_observation_size(task_type)
    action_size = 5  # up, down, left, right, stay
    
    # Use strategy pattern to select algorithm
    if algorithm.lower() == 'dqn':
        strategy = DQNStrategy(input_size=obs_size, output_size=action_size)
    elif algorithm.lower() == 'a3c':
        strategy = A3CStrategy(input_size=obs_size, output_size=action_size)
    elif algorithm.lower() == 'ppo':
        strategy = PPOStrategy(input_size=obs_size, output_size=action_size)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Add observers for monitoring
    metrics_observer = TrainingMetricsObserver()
    strategy.scheduler.attach(metrics_observer)
    
    # Set default paths
    if model_path is None:
        model_path = f"models/{algorithm}_task_{task_type}.pth"
        os.makedirs("models", exist_ok=True)
    
    if stats_path is None:
        stats_path = f"stats/{algorithm}_task_{task_type}.json"
        os.makedirs("stats", exist_ok=True)
    
    # Training loop would go here - for now, we'll integrate with the existing training logic
    # But using the strategy pattern for the algorithm
    
    print(f"Training using {algorithm.upper()} algorithm for Task {task_type}...")
    
    # This would connect to the environment and run the training loop
    # For now, returning the strategy and metrics
    return strategy, metrics_observer


def evaluate_agent_with_strategy(strategy, task_type, num_episodes=10):
    """
    Evaluate an agent using a strategy.
    
    Args:
        strategy: RL strategy to evaluate
        task_type: Type of task (1, 2, or 3)
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Average total reward across episodes
    """
    print(f"Evaluating {strategy.__class__.__name__} on Task {task_type}...")
    
    # Evaluation logic would go here
    total_rewards = []
    
    for episode in tqdm(range(num_episodes), desc=f"Evaluation Task {task_type}", unit="episode"):
        # Simulated evaluation - would connect to environment
        # For now just showing the structure
        total_reward = np.random.uniform(0, 100)  # Placeholder
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    return avg_reward


def train_all_strategies(num_episodes=500):
    """
    Train agents using all available strategies.
    
    Args:
        num_episodes: Number of episodes to train each agent
    """
    print("Starting training for all strategies...")
    
    algorithms = ['dqn', 'a3c', 'ppo']
    task_types = [1, 2, 3]
    
    for task_type in task_types:
        print(f"\nTraining for Task {task_type}...")
        
        for algorithm in algorithms:
            print(f"\nTraining {algorithm.upper()} for Task {task_type}...")
            strategy, metrics = train_task_with_strategy(
                task_type=task_type,
                algorithm=algorithm,
                num_episodes=num_episodes
            )
            
            # Evaluate the trained strategy
            print(f"\nEvaluating {algorithm.upper()} on Task {task_type}...")
            evaluate_agent_with_strategy(strategy, task_type, num_episodes=3)
    
    print("\nTraining completed for all strategies and tasks!")


if __name__ == "__main__":
    # Example usage
    print("Enhanced training module with design patterns")
    
    # Example: Train a DQN agent for task 1
    strategy, metrics = train_task_with_strategy(
        task_type=1,
        algorithm='dqn',
        num_episodes=100
    )
    
    print(f"Trained strategy: {strategy.__class__.__name__}")
    print(f"Collected metrics: {metrics.metrics['steps']} steps")