"""
Unified training system integrating all enhanced features:
- Curriculum learning
- Transfer learning
- Hyperparameter tuning
- Model comparison
- Checkpoint management
- Enhanced monitoring
"""

import numpy as np
import os
from datetime import datetime
import json

from .curriculum_learning import (
    train_with_curriculum, CurriculumScheduler, TransferLearningManager
)
from .hyperparameter_tuning import HyperparameterTuner, ModelComparison
from .enhanced_training_module import (
    train_with_enhanced_monitoring, resume_training_from_checkpoint, compare_training_runs
)
from .training import train_task, evaluate_agent
from .dqn import DQNAgentWrapper


class UnifiedTrainingSystem:
    """
    Main class that unifies all enhanced training features into a single system.
    """
    
    def __init__(self):
        self.transfer_manager = TransferLearningManager()
        self.checkpoint_manager = None
        self.tuners = {}  # Store tuners for each task
        self.comparisons = []  # Store comparisons
    
    def curriculum_training(self, task_type, total_episodes=2000, save_model=True):
        """
        Run curriculum learning for a specific task.
        
        Args:
            task_type: Type of task (1, 2, or 3)
            total_episodes: Total number of episodes to train
            save_model: Whether to save the final model
        
        Returns:
            Trained agent and statistics
        """
        print(f"Starting curriculum learning for Task {task_type}")
        agent, stats = train_with_curriculum(
            task_type=task_type,
            total_episodes=total_episodes,
            save_model=save_model
        )
        return agent, stats
    
    def transfer_learning_training(self, task_sequence, episodes_per_task=500):
        """
        Run transfer learning across a sequence of tasks.
        
        Args:
            task_sequence: List of task types to train on in sequence
            episodes_per_task: Number of episodes per task
        
        Returns:
            Dict mapping task type to trained agent
        """
        print(f"Starting transfer learning across tasks: {task_sequence}")
        agents = self.transfer_manager.multi_task_training(
            task_sequence=task_sequence,
            episodes_per_task=episodes_per_task
        )
        return agents
    
    def hyperparameter_tuning(self, task_type, n_trials=20):
        """
        Run hyperparameter tuning for a specific task.
        
        Args:
            task_type: Type of task (1, 2, or 3)
            n_trials: Number of trials for optimization
        
        Returns:
            Tuner object with best parameters
        """
        print(f"Starting hyperparameter tuning for Task {task_type}")
        tuner = HyperparameterTuner(task_type=task_type, n_trials=n_trials)
        tuner.tune()
        
        # Store tuner for potential later use
        self.tuners[task_type] = tuner
        
        return tuner
    
    def train_with_best_params(self, task_type, num_episodes=1000):
        """
        Train an agent using the best found hyperparameters for a task.
        
        Args:
            task_type: Type of task (1, 2, or 3)
            num_episodes: Number of episodes to train
        
        Returns:
            Trained agent and statistics
        """
        if task_type not in self.tuners:
            print(f"No tuner found for Task {task_type}, running basic tuning first...")
            tuner = self.hyperparameter_tuning(task_type, n_trials=10)
        else:
            tuner = self.tuners[task_type]
        
        if tuner.best_params is None:
            print("No best parameters found, running tuning first...")
            tuner.tune()
        
        agent, stats = tuner.get_best_agent(num_episodes=num_episodes)
        return agent, stats
    
    def enhanced_training_with_monitoring(self, task_type, num_episodes=1000):
        """
        Run enhanced training with detailed monitoring and checkpoints.
        
        Args:
            task_type: Type of task (1, 2, or 3)
            num_episodes: Number of episodes to train
        
        Returns:
            Trained agent and enhanced statistics
        """
        print(f"Starting enhanced training with monitoring for Task {task_type}")
        agent, stats = train_with_enhanced_monitoring(
            task_type=task_type,
            num_episodes=num_episodes,
            save_checkpoints=True,
            checkpoint_freq=100,
            save_model=True
        )
        return agent, stats
    
    def resume_training(self, task_type, checkpoint_path, additional_episodes=500):
        """
        Resume training from a checkpoint.
        
        Args:
            task_type: Type of task (1, 2, or 3)
            checkpoint_path: Path to checkpoint file
            additional_episodes: Number of additional episodes to train
        
        Returns:
            Resumed agent and updated statistics
        """
        print(f"Resuming training for Task {task_type} from checkpoint: {checkpoint_path}")
        agent, stats = resume_training_from_checkpoint(
            task_type=task_type,
            checkpoint_path=checkpoint_path,
            additional_episodes=additional_episodes
        )
        return agent, stats
    
    def compare_models(self, task_type, model_configs):
        """
        Compare different model configurations on the same task.
        
        Args:
            task_type: Type of task (1, 2, or 3)
            model_configs: List of model configurations to compare
        
        Returns:
            Comparison results
        """
        print(f"Comparing models for Task {task_type}")
        comparator = ModelComparison()
        results = comparator.compare_agents(
            agents_config=model_configs,
            task_type=task_type,
            num_episodes=10
        )
        
        self.comparisons.append({
            'task_type': task_type,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
        return results, comparator
    
    def run_comprehensive_training(self, task_type, use_curriculum=True, use_tuning=True, 
                                  num_episodes=1000, save_detailed=True):
        """
        Run a comprehensive training pipeline with multiple enhancements.
        
        Args:
            task_type: Type of task (1, 2, or 3)
            use_curriculum: Whether to use curriculum learning
            use_tuning: Whether to use hyperparameter tuning
            num_episodes: Number of episodes to train
            save_detailed: Whether to save detailed metrics
        
        Returns:
            Final trained agent and comprehensive statistics
        """
        print(f"Starting comprehensive training for Task {task_type}")
        
        if use_tuning:
            print("Step 1: Running hyperparameter tuning...")
            tuner = self.hyperparameter_tuning(task_type, n_trials=15)
            print(f"Best parameters: {tuner.best_params}")
        
        if use_curriculum:
            print("Step 2: Running curriculum learning...")
            agent, stats = self.curriculum_training(task_type, total_episodes=num_episodes//2)
            
            # Optionally save pretrained features for transfer
            self.transfer_manager.save_pretrained_features(agent, task_type)
        else:
            print("Step 2: Running standard training...")
            if use_tuning and task_type in self.tuners:
                # Use tuned parameters
                agent, stats = self.train_with_best_params(task_type, num_episodes//2)
            else:
                # Use standard training
                agent, stats = train_task(task_type, num_episodes=num_episodes//2, save_model=False)
        
        print("Step 3: Running enhanced training with detailed monitoring...")
        # Continue training with enhanced monitoring
        agent, enhanced_stats = train_with_enhanced_monitoring(
            task_type=task_type,
            num_episodes=num_episodes//2,
            save_checkpoints=True,
            checkpoint_freq=50,
            save_model=True,
            log_detailed_metrics=save_detailed
        )
        
        # Combine stats
        if hasattr(stats, 'episode_rewards'):
            # If we have curriculum stats, combine them
            enhanced_stats.episode_rewards = stats.episode_rewards + enhanced_stats.episode_rewards
            enhanced_stats.episode_lengths = stats.episode_lengths + enhanced_stats.episode_lengths
            enhanced_stats.episode_successes = stats.episode_successes + enhanced_stats.episode_successes
        
        print(f"Comprehensive training completed for Task {task_type}")
        return agent, enhanced_stats
    
    def save_system_state(self, filepath=None):
        """
        Save the entire system state including tuners and comparison results.
        
        Args:
            filepath: Path to save system state
        """
        if filepath is None:
            filepath = f"system_state/system_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("system_state", exist_ok=True)
        
        system_state = {
            'timestamp': datetime.now().isoformat(),
            'tuned_tasks': list(self.tuners.keys()),
            'comparisons_made': len(self.comparisons),
            'transfer_sessions': len(self.transfer_manager.pretrained_models)
        }
        
        # Save best parameters for each task
        best_params = {}
        for task_type, tuner in self.tuners.items():
            if tuner.best_params:
                best_params[task_type] = tuner.best_params
        
        system_state['best_params'] = best_params
        
        with open(filepath, 'w') as f:
            json.dump(system_state, f, indent=2)
        
        print(f"System state saved to {filepath}")
    
    def run_complete_pipeline(self, task_types=[1, 2, 3], num_episodes=1500):
        """
        Run the complete enhanced training pipeline for multiple tasks.
        
        Args:
            task_types: List of task types to train
            num_episodes: Number of episodes per task
        
        Returns:
            Dict mapping task type to (agent, stats)
        """
        print("Starting complete enhanced training pipeline...")
        results = {}
        
        for task_type in task_types:
            print(f"\n--- Processing Task {task_type} ---")
            try:
                agent, stats = self.run_comprehensive_training(
                    task_type=task_type,
                    use_curriculum=True,
                    use_tuning=True,
                    num_episodes=num_episodes,
                    save_detailed=True
                )
                results[task_type] = (agent, stats)
                
                # Evaluate the final agent
                print(f"Evaluating final agent for Task {task_type}...")
                avg_reward = evaluate_agent(agent, task_type, num_episodes=10)
                print(f"Final evaluation - Task {task_type}: Avg Reward = {avg_reward:.2f}")
                
            except Exception as e:
                print(f"Error processing Task {task_type}: {e}")
                continue
        
        # Save system state
        self.save_system_state()
        
        print("\nComplete enhanced training pipeline finished!")
        return results


def run_demo():
    """
    Run a demonstration of the unified training system.
    """
    print("Demonstration of Unified Training System")
    print("=" * 50)
    
    # Initialize the system
    system = UnifiedTrainingSystem()
    
    # Example 1: Run curriculum learning for Task 1
    print("\n1. Curriculum Learning Demo:")
    agent1, stats1 = system.curriculum_training(task_type=1, total_episodes=500)
    
    # Example 2: Run hyperparameter tuning for Task 2
    print("\n2. Hyperparameter Tuning Demo:")
    tuner = system.hyperparameter_tuning(task_type=2, n_trials=10)
    
    # Example 3: Enhanced training with monitoring for Task 3
    print("\n3. Enhanced Training with Monitoring Demo:")
    agent3, stats3 = system.enhanced_training_with_monitoring(task_type=3, num_episodes=300)
    
    # Example 4: Transfer learning between tasks
    print("\n4. Transfer Learning Demo:")
    agents = system.transfer_learning_training(task_sequence=[1, 2], episodes_per_task=200)
    
    # Example 5: Model comparison
    print("\n5. Model Comparison Demo:")
    model_configs = [
        (DQNAgentWrapper, {'lr': 0.001, 'batch_size': 32}, "Standard DQN"),
        (DQNAgentWrapper, {'lr': 0.0005, 'batch_size': 64}, "Conservative DQN")
    ]
    comparison_results, comparator = system.compare_models(2, model_configs)
    print(comparator.get_comparison_report())
    
    print("\nDemo completed!")


if __name__ == "__main__":
    # Run the demo
    run_demo()