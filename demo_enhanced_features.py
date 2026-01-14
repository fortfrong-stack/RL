#!/usr/bin/env python3
"""
Demo script showcasing all the enhanced features implemented:
- Curriculum learning
- Transfer learning
- Hyperparameter tuning
- Model comparison
- Checkpoint management
- Enhanced monitoring
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl.unified_training_system import UnifiedTrainingSystem, run_demo
from rl.curriculum_learning import CurriculumScheduler
from rl.hyperparameter_tuning import HyperparameterTuner, ModelComparison
from rl.enhanced_training_module import train_with_enhanced_monitoring, CheckpointManager


def demo_curriculum_learning():
    """Demonstrate curriculum learning."""
    print("\n" + "="*60)
    print("DEMO: Curriculum Learning")
    print("="*60)
    
    system = UnifiedTrainingSystem()
    
    # Demonstrate curriculum learning for Task 1
    print("\nStarting curriculum learning for Task 1 (Find all sources)...")
    agent, stats = system.curriculum_training(task_type=1, total_episodes=300)
    
    print(f"Completed curriculum learning. Final stats:")
    print(f"- Episodes completed: {len(stats.episode_rewards)}")
    print(f"- Average reward: {sum(stats.episode_rewards)/len(stats.episode_rewards):.2f}")
    print(f"- Success rate: {sum(stats.episode_successes)/len(stats.episode_successes):.2f}")


def demo_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning."""
    print("\n" + "="*60)
    print("DEMO: Hyperparameter Tuning")
    print("="*60)
    
    system = UnifiedTrainingSystem()
    
    print("\nStarting hyperparameter tuning for Task 2 (Find quietest place)...")
    tuner = system.hyperparameter_tuning(task_type=2, n_trials=10)  # Reduced trials for demo
    
    print(f"Best parameters found: {tuner.best_params}")
    print(f"Best score achieved: {tuner.study.best_value:.2f}")


def demo_enhanced_monitoring():
    """Demonstrate enhanced monitoring and checkpointing."""
    print("\n" + "="*60)
    print("DEMO: Enhanced Monitoring & Checkpointing")
    print("="*60)
    
    system = UnifiedTrainingSystem()
    
    print("\nStarting enhanced training with monitoring for Task 3...")
    agent, stats = system.enhanced_training_with_monitoring(task_type=3, num_episodes=200)
    
    print(f"Training completed with {len(stats.episode_rewards)} episodes")
    print(f"Checkpoints saved in 'checkpoints/' directory")
    print(f"Models saved in 'models/' directory")


def demo_model_comparison():
    """Demonstrate model comparison."""
    print("\n" + "="*60)
    print("DEMO: Model Comparison")
    print("="*60)
    
    system = UnifiedTrainingSystem()
    
    # Define different model configurations to compare
    model_configs = [
        ('Standard DQN', {'lr': 0.001, 'gamma': 0.99, 'batch_size': 32}),
        ('Conservative DQN', {'lr': 0.0005, 'gamma': 0.95, 'batch_size': 64}),
        ('Aggressive DQN', {'lr': 0.002, 'gamma': 0.999, 'batch_size': 16})
    ]
    
    print("\nComparing different DQN configurations for Task 1...")
    
    # Convert to the format expected by the comparison function
    formatted_configs = []
    from rl.dqn import DQNAgentWrapper
    for name, params in model_configs:
        formatted_configs.append((DQNAgentWrapper, params, name))
    
    results, comparator = system.compare_models(1, formatted_configs)
    
    print("\nComparison Results:")
    print(comparator.get_comparison_report())


def demo_transfer_learning():
    """Demonstrate transfer learning."""
    print("\n" + "="*60)
    print("DEMO: Transfer Learning")
    print("="*60)
    
    system = UnifiedTrainingSystem()
    
    print("\nStarting transfer learning across tasks [1, 2]...")
    agents = system.transfer_learning_training(task_sequence=[1, 2], episodes_per_task=150)
    
    print(f"Trained agents for tasks: {list(agents.keys())}")
    print("Transfer learning completed successfully.")


def demo_comprehensive_pipeline():
    """Demonstrate the comprehensive training pipeline."""
    print("\n" + "="*60)
    print("DEMO: Comprehensive Training Pipeline")
    print("="*60)
    
    system = UnifiedTrainingSystem()
    
    print("\nStarting comprehensive training for Task 1...")
    agent, stats = system.run_comprehensive_training(
        task_type=1,
        use_curriculum=True,
        use_tuning=True,
        num_episodes=400,
        save_detailed=True
    )
    
    print(f"Comprehensive training completed!")
    print(f"- Used curriculum learning: True")
    print(f"- Used hyperparameter tuning: True") 
    print(f"- Used enhanced monitoring: True")
    print(f"- Total episodes: {len(stats.episode_rewards)}")


def main():
    """Run all demos."""
    print("Enhanced RL Training Features Demo")
    print("="*60)
    
    print("\nThis demo showcases the following enhanced features:")
    print("1. Curriculum Learning - Gradual increase in difficulty")
    print("2. Transfer Learning - Knowledge transfer between tasks") 
    print("3. Hyperparameter Tuning - Automatic optimization")
    print("4. Enhanced Monitoring - Detailed metrics and logging")
    print("5. Checkpoint Management - Save/restore training progress")
    print("6. Model Comparison - Compare different configurations")
    
    # Run individual demos
    try:
        demo_curriculum_learning()
        demo_hyperparameter_tuning()
        demo_enhanced_monitoring()
        demo_model_comparison()
        demo_transfer_learning()
        demo_comprehensive_pipeline()
    except Exception as e:
        print(f"An error occurred during the demo: {e}")
        print("This may be due to missing dependencies or resource constraints.")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nThe enhanced training system provides:")
    print("- Automated curriculum learning for gradual skill building")
    print("- Intelligent hyperparameter optimization")
    print("- Comprehensive monitoring and logging")
    print("- Checkpoint management for long training runs")
    print("- Model comparison tools")
    print("- Transfer learning capabilities")


if __name__ == "__main__":
    main()