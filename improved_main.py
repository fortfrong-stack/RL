"""
Improved main entry point for the sound-based navigation system.
Demonstrates the implemented design patterns: Factory, Strategy, Observer.
"""

import argparse
import sys
import os
from typing import Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patterns.factory.rl_factory import DQNFactory, A3CFactory, PPOFactory, EnvironmentFactory, VisualizationFactory
from patterns.strategy.rl_strategies import DQNStrategy, A3CStrategy, PPOStrategy, TrainingMetricsObserver, EventLoggingObserver
from patterns.observer.visualization_observer import ImprovedPygameVisualizer, EventLoggerObserver
from patterns.factory.serialization_manager import SerializationManager
from utils.environment_gen import generate_random_environment, manual_environment_setup
from rl.training import train_task, evaluate_agent, train_all_tasks
from interface.console_ui import main_menu


def get_observation_size(task_type: int) -> int:
    """
    Get the size of the observation vector for a given task type.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        
    Returns:
        Size of the observation vector
    """
    from utils.audio_processing import get_audio_observation_features
    sample_obs = get_audio_observation_features(0.5, 0.5)
    return len(sample_obs)


def main():
    """Main function to handle command-line arguments and execute the appropriate mode."""
    parser = argparse.ArgumentParser(description='Sound-Based Navigation System with Design Patterns')
    parser.add_argument('--mode', choices=['train', 'test', 'demo'], required=True,
                        help='Mode: train, test, or demo')
    parser.add_argument('--task', type=int, choices=[1, 2, 3], required=True,
                        help='Task type: 1-Find all sources, 2-Find quietest place, 3-Follow moving source')
    parser.add_argument('--algorithm', choices=['dqn', 'a3c', 'ppo'], default='dqn',
                        help='RL algorithm to use: dqn, a3c, or ppo')
    parser.add_argument('--setup', choices=['random', 'manual'], default='random',
                        help='Environment setup method')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training (default: 1000)')
    parser.add_argument('--model-path', type=str,
                        help='Path to save/load model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"Training agent for Task {args.task} using {args.algorithm.upper()} algorithm...")
        
        # Use factory pattern to create the appropriate agent
        factory_map = {
            'dqn': DQNFactory(),
            'a3c': A3CFactory(),
            'ppo': PPOFactory()
        }
        
        obs_size = get_observation_size(args.task)
        action_size = 5  # up, down, left, right, stay
        
        agent = factory_map[args.algorithm].create_agent(
            input_size=obs_size,
            output_size=action_size
        )
        
        # Use strategy pattern to wrap the agent in a strategy
        strategy_map = {
            'dqn': DQNStrategy(input_size=obs_size, output_size=action_size),
            'a3c': A3CStrategy(input_size=obs_size, output_size=action_size),
            'ppo': PPOStrategy(input_size=obs_size, output_size=action_size)
        }
        
        strategy = strategy_map[args.algorithm]
        
        # Add observers for the strategy
        metrics_observer = TrainingMetricsObserver()
        logging_observer = EventLoggingObserver()
        strategy.scheduler.attach(metrics_observer)
        strategy.scheduler.attach(logging_observer)
        
        # Train using the existing training function
        trained_agent, stats = train_task(
            task_type=args.task,
            num_episodes=args.episodes,
            model_path=args.model_path
        )
        
        # Save model using improved serialization
        if args.model_path:
            SerializationManager.save_model(trained_agent, args.model_path, {
                'task_type': args.task,
                'algorithm': args.algorithm,
                'episodes': args.episodes
            })
        
        print(f"Evaluating trained agent for Task {args.task}...")
        evaluate_agent(trained_agent, args.task, num_episodes=5)
        
        # Print metrics collected by observer
        print(f"Training metrics - Steps: {metrics_observer.metrics['steps']}, "
              f"Average loss: {sum(metrics_observer.metrics['losses']) / len(metrics_observer.metrics['losses']) if metrics_observer.metrics['losses'] else 0:.4f}")
        
    elif args.mode == 'test':
        print("Launching test interface...")
        main_menu()
        
    elif args.mode == 'demo':
        print("Running demonstration of design patterns...")
        
        # Demo Factory Pattern
        print("\n--- Factory Pattern Demo ---")
        env_factory = EnvironmentFactory()
        env = env_factory.create_environment(args.task)
        print(f"Created environment for Task {args.task} using factory")
        
        viz_factory = VisualizationFactory()
        viz = viz_factory.create_visualizer('pygame')
        print("Created Pygame visualizer using factory")
        
        # Demo Strategy Pattern
        print("\n--- Strategy Pattern Demo ---")
        obs_size = get_observation_size(args.task)
        strategies = [
            DQNStrategy(input_size=obs_size, output_size=5),
            A3CStrategy(input_size=obs_size, output_size=5),
            PPOStrategy(input_size=obs_size, output_size=5)
        ]
        
        for i, strategy in enumerate(['DQN', 'A3C', 'PPO']):
            print(f"Using {strategy} strategy")
        
        # Demo Observer Pattern
        print("\n--- Observer Pattern Demo ---")
        improved_viz = ImprovedPygameVisualizer()
        logger = EventLoggerObserver()
        improved_viz.subject.attach(logger)
        print("Attached event logger to visualizer")
        
        # Clean up observers
        improved_viz.subject.detach(logger)
        
        print("\nDemo completed!")


if __name__ == "__main__":
    # If no command line arguments, launch the main menu
    if len(sys.argv) == 1:
        main_menu()
    else:
        main()