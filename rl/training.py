"""
Training module for DQN agents on sound-based navigation tasks.
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


def train_task(task_type, num_episodes=1000, save_model=True, model_path=None):
    """
    Train a DQN agent for a specific task.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        num_episodes: Number of episodes to train
        save_model: Whether to save the trained model
        model_path: Path to save the model (default: models/dqn_task_{task_type}.pth)
    
    Returns:
        Trained DQNAgentWrapper
    """
    # Get observation size
    obs_size = get_observation_size(task_type)
    action_size = 5  # up, down, left, right, stay
    
    # Initialize agent
    agent = DQNAgentWrapper(
        input_size=obs_size,
        output_size=action_size,
        lr=0.001 if task_type == 1 else (0.0005 if task_type == 2 else 0.0008),  # Different learning rates
        epsilon_decay=0.99 if task_type == 1 else (0.95 if task_type == 2 else 0.98),  # Different epsilon decays
    )
    
    # Set default model path
    if model_path is None:
        model_path = f"models/dqn_task_{task_type}.pth"
        os.makedirs("models", exist_ok=True)
    
    # Training loop
    for episode in range(num_episodes):
        # Generate a random environment for this episode
        env = generate_random_environment(task_type)
        
        # Get initial observation
        state = env.reset()
        
        total_reward = 0
        step_count = 0
        
        while not env.done and step_count < env.max_steps:
            # Select action using epsilon-greedy
            action = agent.act(state, training=True)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent
            agent.replay()
            
            # Update state and tracking variables
            state = next_state
            total_reward += reward
            step_count += 1
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f"Task {task_type}, Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Steps: {step_count}")
    
    # Save the model if requested
    if save_model:
        agent.save(model_path)
        print(f"Model saved to {model_path}")
    
    return agent


def evaluate_agent(agent, task_type, num_episodes=10, render=False):
    """
    Evaluate a trained agent on a task.
    
    Args:
        agent: Trained DQNAgentWrapper
        task_type: Type of task (1, 2, or 3)
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment (not implemented yet)
    
    Returns:
        Average total reward across episodes
    """
    total_rewards = []
    
    for episode in range(num_episodes):
        # Generate a random environment for evaluation
        env = generate_random_environment(task_type)
        
        # Get initial observation
        state = env.reset()
        
        total_reward = 0
        step_count = 0
        
        while not env.done and step_count < env.max_steps:
            # Select action (no exploration during evaluation)
            action = agent.act(state, training=False)
            
            # Take action in environment
            state, reward, done = env.step(action)
            
            total_reward += reward
            step_count += 1
        
        total_rewards.append(total_reward)
        print(f"Evaluation - Episode {episode + 1}, Total Reward: {total_reward:.2f}, Steps: {step_count}")
    
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
    
    for task_type in [1, 2, 3]:
        print(f"\nTraining for Task {task_type}...")
        agent = train_task(task_type, num_episodes=num_episodes)
        
        # Evaluate the trained agent
        print(f"\nEvaluating Task {task_type} agent...")
        evaluate_agent(agent, task_type, num_episodes=5)
    
    print("\nTraining completed for all tasks!")