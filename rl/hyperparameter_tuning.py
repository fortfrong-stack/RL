"""
Hyperparameter tuning implementation for automatic optimization.
"""

import numpy as np
import optuna
import os
from datetime import datetime
import json
import copy

from .training import train_task, evaluate_agent
from .dqn import DQNAgentWrapper


class HyperparameterTuner:
    """
    Class for hyperparameter tuning using Optuna.
    """
    
    def __init__(self, task_type, objective_func=None, n_trials=50):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            task_type: Type of task (1, 2, or 3)
            objective_func: Custom objective function for optimization
            n_trials: Number of trials for optimization
        """
        self.task_type = task_type
        self.n_trials = n_trials
        self.study = optuna.create_study(direction='maximize')
        self.best_params = None
        self.trial_results = []
        
        if objective_func is None:
            self.objective_func = self.default_objective
        else:
            self.objective_func = objective_func
    
    def default_objective(self, trial, params):
        """
        Default objective function for hyperparameter tuning.
        Optimizes for average reward over a few evaluation episodes.
        """
        # Train agent with given parameters
        obs_size = self.get_observation_size(self.task_type)
        agent = DQNAgentWrapper(
            input_size=obs_size,
            output_size=5,
            lr=params.get('lr', 0.001),
            gamma=params.get('gamma', 0.99),
            epsilon=params.get('epsilon', 1.0),
            epsilon_decay=params.get('epsilon_decay', 0.995),
            epsilon_min=params.get('epsilon_min', 0.01),
            target_update_freq=params.get('target_update_freq', 100),
            batch_size=params.get('batch_size', 32),
            buffer_size=params.get('buffer_size', 10000)
        )
        
        # Short training run to get preliminary results
        from core.tasks import create_task_environment
        from tqdm import tqdm
        
        # Train for fewer episodes to speed up tuning
        eval_episodes = 50  # Short training for tuning purposes
        
        for episode in tqdm(range(eval_episodes), desc=f"Tuning Trial", leave=False):
            env = create_task_environment(self.task_type)
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
        
        # Evaluate the agent
        avg_reward = evaluate_agent(agent, self.task_type, num_episodes=5)
        
        return avg_reward
    
    def get_observation_size(self, task_type):
        """Get the size of the observation vector for a given task type."""
        from utils.audio_processing import get_audio_observation_features
        sample_obs = get_audio_observation_features(0.5, 0.5)
        return len(sample_obs)
    
    def tune(self, timeout=None):
        """
        Run hyperparameter tuning.
        
        Args:
            timeout: Timeout in seconds for the tuning process
        """
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
                'gamma': trial.suggest_float('gamma', 0.8, 0.999),
                'epsilon': trial.suggest_float('epsilon', 0.5, 1.0),
                'epsilon_decay': trial.suggest_float('epsilon_decay', 0.9, 0.999),
                'epsilon_min': trial.suggest_float('epsilon_min', 0.001, 0.1),
                'target_update_freq': trial.suggest_int('target_update_freq', 50, 300),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'buffer_size': trial.suggest_categorical('buffer_size', [5000, 10000, 20000, 50000])
            }
            
            # Add hidden_size parameter
            params['hidden_size'] = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
            
            # Store params in trial for later access
            trial.set_user_attr('params', params)
            
            # Evaluate the parameters
            score = self.objective_func(trial, params)
            
            # Store trial results
            self.trial_results.append({
                'trial_number': trial.number,
                'score': score,
                'params': copy.deepcopy(params)
            })
            
            return score
        
        # Run optimization
        self.study.optimize(objective, n_trials=self.n_trials, timeout=timeout)
        
        # Store best parameters
        self.best_params = self.study.best_trial.user_attrs['params']
        
        print(f"Best parameters found: {self.best_params}")
        print(f"Best score: {self.study.best_value}")
        
        return self.best_params
    
    def get_best_agent(self, num_episodes=1000):
        """
        Train an agent with the best found hyperparameters.
        
        Args:
            num_episodes: Number of episodes to train the final agent
        
        Returns:
            Trained agent with best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("Must run tune() first to find best parameters")
        
        obs_size = self.get_observation_size(self.task_type)
        agent = DQNAgentWrapper(
            input_size=obs_size,
            output_size=5,
            lr=self.best_params['lr'],
            gamma=self.best_params['gamma'],
            epsilon=self.best_params['epsilon'],
            epsilon_decay=self.best_params['epsilon_decay'],
            epsilon_min=self.best_params['epsilon_min'],
            target_update_freq=self.best_params['target_update_freq'],
            batch_size=self.best_params['batch_size'],
            buffer_size=self.best_params['buffer_size']
        )
        
        # Train the agent with best parameters
        agent, stats = train_task(
            task_type=self.task_type,
            num_episodes=num_episodes,
            save_model=False
        )
        
        return agent, stats
    
    def save_results(self, filepath=None):
        """
        Save tuning results to a file.
        
        Args:
            filepath: Path to save results
        """
        if filepath is None:
            filepath = f"tuning_results/task_{self.task_type}_tuning_results.json"
            os.makedirs("tuning_results", exist_ok=True)
        
        results = {
            'task_type': self.task_type,
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': self.n_trials,
            'trial_results': self.trial_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Tuning results saved to {filepath}")


class ModelComparison:
    """
    Class for comparing performance of different models.
    """
    
    def __init__(self):
        self.comparison_results = []
    
    def compare_agents(self, agents_config, task_type, num_episodes=10):
        """
        Compare multiple agents on the same task.
        
        Args:
            agents_config: List of tuples (agent_class, params, name) or pre-trained agents
            task_type: Type of task to evaluate on
            num_episodes: Number of episodes for evaluation
        
        Returns:
            Comparison results
        """
        results = []
        
        for config in agents_config:
            if isinstance(config, tuple) and len(config) == 3:
                # Config is (class, params, name) - create and train agent
                agent_class, params, name = config
                agent = self._create_agent(agent_class, params, task_type)
                
                # Train the agent briefly if needed
                if hasattr(agent, 'q_network'):  # It's a DQN agent
                    from core.tasks import create_task_environment
                    from tqdm import tqdm
                    
                    # Brief training to make sure agent is functional
                    for episode in tqdm(range(10), desc=f"Initial training for {name}", leave=False):
                        env = create_task_environment(task_type)
                        state = env.reset()
                        
                        while not env.done:
                            action = agent.act(state, training=True)
                            next_state, reward, done = env.step(action)
                            
                            agent.remember(state, action, reward, next_state, done)
                            
                            if len(agent.memory) > agent.batch_size:
                                agent.replay()
                            
                            state = next_state
                
                avg_reward = evaluate_agent(agent, task_type, num_episodes=num_episodes)
            else:
                # Config is a pre-trained agent or other object
                agent = config
                if hasattr(agent, '__call__'):  # It's a function that creates an agent
                    agent = agent()
                
                avg_reward = evaluate_agent(agent, task_type, num_episodes=num_episodes)
                name = getattr(agent, 'name', f"Agent_{len(results)+1}")
            
            result = {
                'name': name,
                'avg_reward': avg_reward,
                'task_type': task_type
            }
            results.append(result)
        
        self.comparison_results.extend(results)
        return results
    
    def _create_agent(self, agent_class, params, task_type):
        """Create an agent with given parameters."""
        from .dqn import get_observation_size
        
        obs_size = get_observation_size(task_type)
        
        if agent_class == DQNAgentWrapper:
            return DQNAgentWrapper(input_size=obs_size, output_size=5, **params)
        else:
            # Assume it's a custom agent class
            return agent_class(obs_size, 5, **params)
    
    def get_comparison_report(self):
        """Generate a comparison report."""
        if not self.comparison_results:
            return "No comparison results available."
        
        report = "Model Comparison Report\n"
        report += "=" * 50 + "\n\n"
        
        # Sort by average reward
        sorted_results = sorted(self.comparison_results, key=lambda x: x['avg_reward'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            report += f"{i}. {result['name']}: {result['avg_reward']:.2f} avg reward\n"
        
        report += f"\nBest performing model: {sorted_results[0]['name']} ({sorted_results[0]['avg_reward']:.2f})"
        
        return report
    
    def save_comparison(self, filepath=None):
        """Save comparison results to a file."""
        if filepath is None:
            filepath = f"comparison_results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("comparison_results", exist_ok=True)
        
        comparison_data = {
            'results': self.comparison_results,
            'timestamp': datetime.now().isoformat(),
            'report': self.get_comparison_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"Comparison results saved to {filepath}")


def run_hyperparameter_tuning_example():
    """Example of how to use the hyperparameter tuner."""
    print("Running hyperparameter tuning example...")
    
    # Create tuner for task 1
    tuner = HyperparameterTuner(task_type=1, n_trials=10)  # Using fewer trials for demo
    
    # Run tuning
    best_params = tuner.tune(timeout=300)  # 5 minutes timeout
    
    # Save results
    tuner.save_results()
    
    return tuner


def run_model_comparison_example():
    """Example of how to compare different models."""
    print("Running model comparison example...")
    
    # Define different configurations to compare
    configs = [
        (DQNAgentWrapper, {'lr': 0.001, 'gamma': 0.99, 'batch_size': 32}, "Standard DQN"),
        (DQNAgentWrapper, {'lr': 0.0005, 'gamma': 0.95, 'batch_size': 64}, "Slow Learner DQN"),
        (DQNAgentWrapper, {'lr': 0.002, 'gamma': 0.999, 'batch_size': 16}, "Fast Learner DQN")
    ]
    
    # Create comparator
    comparator = ModelComparison()
    
    # Compare agents on task 1
    results = comparator.compare_agents(configs, task_type=1, num_episodes=5)
    
    # Print report
    print(comparator.get_comparison_report())
    
    # Save results
    comparator.save_comparison()
    
    return comparator


if __name__ == "__main__":
    # Example usage
    print("Hyperparameter tuning and model comparison examples")
    
    # Example of hyperparameter tuning
    # tuner = run_hyperparameter_tuning_example()
    
    # Example of model comparison
    # comparator = run_model_comparison_example()