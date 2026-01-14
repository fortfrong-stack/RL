"""
Main entry point for the sound-based navigation system.
"""

import argparse
import sys
import os
from typing import Optional

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.environment_gen import generate_random_environment, manual_environment_setup
from rl.unified_training_system import UnifiedTrainingSystem
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
    parser = argparse.ArgumentParser(description='Sound-Based Navigation System')
    parser.add_argument('--mode', choices=['train', 'test', 'enhanced'], required=True,
                        help='Mode: train, test, or enhanced')
    parser.add_argument('--task', type=int, choices=[1, 2, 3], required=True,
                        help='Task type: 1-Find all sources, 2-Find quietest place, 3-Follow moving source')
    parser.add_argument('--setup', choices=['random', 'manual'], default='random',
                        help='Environment setup method')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training (default: 1000)')
    parser.add_argument('--model-path', type=str,
                        help='Path to save/load model')
    parser.add_argument('--enhanced-method', choices=['curriculum', 'tuning', 'monitoring', 'transfer', 'all'],
                        help='Enhanced training method to use (for enhanced mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"Training agent for Task {args.task}...")
        agent, _ = train_task(
            task_type=args.task,
            num_episodes=args.episodes,
            model_path=args.model_path
        )
        
        print(f"Evaluating trained agent for Task {args.task}...")
        evaluate_agent(agent, args.task, num_episodes=5)
        
    elif args.mode == 'enhanced':
        print(f"Running enhanced training for Task {args.task}...")
        system = UnifiedTrainingSystem()
        
        if args.enhanced_method == 'curriculum':
            agent, stats = system.curriculum_training(
                task_type=args.task,
                total_episodes=args.episodes
            )
        elif args.enhanced_method == 'tuning':
            tuner = system.hyperparameter_tuning(args.task, n_trials=min(20, args.episodes//50))
            agent, stats = system.train_with_best_params(args.task, num_episodes=args.episodes)
        elif args.enhanced_method == 'monitoring':
            agent, stats = system.enhanced_training_with_monitoring(
                task_type=args.task,
                num_episodes=args.episodes
            )
        elif args.enhanced_method == 'transfer':
            # For transfer learning, we need a sequence of tasks
            task_sequence = [args.task]  # We can extend this to include related tasks
            agents = system.transfer_learning_training(task_sequence, episodes_per_task=args.episodes)
            agent = agents[args.task]
            # Note: For transfer learning, we might want to define related tasks
        elif args.enhanced_method == 'all':
            agent, stats = system.run_comprehensive_training(
                task_type=args.task,
                use_curriculum=True,
                use_tuning=True,
                num_episodes=args.episodes,
                save_detailed=True
            )
        else:
            # Default to comprehensive training if no specific method is chosen
            agent, stats = system.run_comprehensive_training(
                task_type=args.task,
                use_curriculum=True,
                use_tuning=True,
                num_episodes=args.episodes,
                save_detailed=True
            )
        
        print(f"Evaluating enhanced trained agent for Task {args.task}...")
        evaluate_agent(agent, args.task, num_episodes=5)
        
    elif args.mode == 'test':
        print("Launching test interface...")
        main_menu()


if __name__ == "__main__":
    # If no command line arguments, launch the main menu
    if len(sys.argv) == 1:
        main_menu()
    else:
        main()