"""
Main entry point for the sound-based navigation system.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.environment_gen import generate_random_environment, manual_environment_setup
from rl.dqn import DQNAgentWrapper
from rl.training import train_task, evaluate_agent, train_all_tasks
from interface.console_ui import test_interface
from utils.visualization import PygameVisualizer


def get_observation_size(task_type):
    """Get the size of the observation vector for a given task type."""
    from utils.audio_processing import get_audio_observation_features
    sample_obs = get_audio_observation_features(0.5, 0.5)
    return len(sample_obs)


def main():
    parser = argparse.ArgumentParser(description='Sound-Based Navigation System')
    parser.add_argument('--mode', choices=['train', 'test'], required=True,
                        help='Mode: train or test')
    parser.add_argument('--task', type=int, choices=[1, 2, 3], required=True,
                        help='Task type: 1-Find all sources, 2-Find quietest place, 3-Follow moving source')
    parser.add_argument('--setup', choices=['random', 'manual'], default='random',
                        help='Environment setup method')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training (default: 1000)')
    parser.add_argument('--model-path', type=str,
                        help='Path to save/load model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"Training agent for Task {args.task}...")
        agent = train_task(
            task_type=args.task,
            num_episodes=args.episodes,
            model_path=args.model_path
        )
        
        print(f"Evaluating trained agent for Task {args.task}...")
        evaluate_agent(agent, args.task, num_episodes=5)
        
    elif args.mode == 'test':
        print("Launching test interface...")
        test_interface()


if __name__ == "__main__":
    # If no command line arguments, launch the main menu
    if len(sys.argv) == 1:
        from interface.console_ui import main_menu
        main_menu()
    else:
        main()