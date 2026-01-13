"""
Console-based user interface for testing and interacting with the sound navigation environment.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

try:
    from ..utils.environment_gen import generate_random_environment, manual_environment_setup
    from ..rl.dqn import DQNAgentWrapper
    from ..utils.visualization import PygameVisualizer
    from ..core.tasks import create_task_environment
except ImportError:
    from utils.environment_gen import generate_random_environment, manual_environment_setup
    from rl.dqn import DQNAgentWrapper
    from utils.visualization import PygameVisualizer
    from core.tasks import create_task_environment


def get_observation_size(task_type):
    """
    Get the size of the observation vector for a given task type.
    """
    from utils.audio_processing import get_audio_observation_features
    sample_obs = get_audio_observation_features(0.5, 0.5)
    return len(sample_obs)


def load_trained_agent(task_type, model_path=None):
    """
    Load a trained agent from a saved model file.
    
    Args:
        task_type: Type of task (1, 2, or 3)
        model_path: Path to the saved model file
    
    Returns:
        Loaded DQNAgentWrapper
    """
    try:
        from rl.dqn import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            print("PyTorch is not available. Cannot load trained agent.")
            return None
    except ImportError:
        print("PyTorch is not available. Cannot load trained agent.")
        return None
    
    if model_path is None:
        model_path = f"models/dqn_task_{task_type}.pth"
    
    obs_size = get_observation_size(task_type)
    action_size = 5
    
    agent = DQNAgentWrapper(input_size=obs_size, output_size=action_size)
    agent.load(model_path)
    
    return agent


def test_interface():
    """
    Console interface for testing the environment and trained agents.
    """
    print("=== Sound Navigation Environment Test Interface ===")
    print("Select a task:")
    print("1 - Find all sound sources")
    print("2 - Find the quietest place")
    print("3 - Follow moving sound source")
    
    try:
        task = int(input("Your choice (1-3): "))
        if task not in [1, 2, 3]:
            print("Invalid choice. Exiting.")
            return
    except ValueError:
        print("Invalid input. Exiting.")
        return
    
    print("\nSelect mode:")
    print("(r)andom - Randomly generated environment")
    print("(m)anual - Manually configured environment")
    
    mode = input("Your choice (r/m): ").lower()
    
    if mode == 'm':
        # Manual environment setup
        try:
            env = manual_environment_setup(task)
        except Exception as e:
            print(f"Error in manual setup: {e}")
            return
    else:
        # Random environment
        try:
            env = generate_random_environment(task)
        except Exception as e:
            print(f"Error in random setup: {e}")
            return
    
    print(f"\nEnvironment created for Task {task}. Starting simulation...")
    print("Controls: w(up), s(down), a(left), d(right), x(stay), q(quit)")
    
    # Ask if user wants to use trained agent or manual control
    agent_choice = input("\nUse trained agent? (y/n): ").lower()
    
    if agent_choice == 'y':
        try:
            agent = load_trained_agent(task)
            print("Loaded trained agent.")
        except Exception as e:
            print(f"Could not load trained agent: {e}")
            print("Switching to manual control.")
            agent = None
    else:
        agent = None
    
    # Initialize visualization
    viz = PygameVisualizer()
    
    # Main loop
    step = 0
    total_reward = 0
    
    # Reset environment
    obs = env.reset()
    
    running = True
    while running and not env.done and step < env.max_steps:
        # Update visualization
        viz.update(env)
        
        print(f"\nStep: {step}, Position: {env.agent.get_position()}, "
              f"Reward: {env.total_reward:.2f}")
        
        if agent:
            # Use trained agent
            action = agent.act(obs, training=False)
            print(f"Agent chose action: {action} ({get_action_name(action)})")
        else:
            # Manual control
            key = input("Enter action (w/a/s/d/x/q): ").lower()
            
            if key == 'q':
                break
            elif key == 'w':
                action = 0  # up
            elif key == 's':
                action = 1  # down
            elif key == 'a':
                action = 2  # left
            elif key == 'd':
                action = 3  # right
            elif key == 'x':
                action = 4  # stay
            else:
                print("Invalid key. Using stay action.")
                action = 4
        
        # Take action
        obs, reward, done = env.step(action)
        total_reward += reward
        step += 1
        
        if done:
            print(f"Episode finished! Total reward: {total_reward:.2f}, Steps: {step}")
            break
    
    viz.close()
    print("Simulation ended.")


def get_action_name(action):
    """
    Get the name of an action for display purposes.
    
    Args:
        action: Action index (0-4)
    
    Returns:
        Name of the action
    """
    names = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    return names[action] if 0 <= action < len(names) else "UNKNOWN"


def main_menu():
    """
    Main menu for the console interface.
    """
    while True:
        print("\n=== Sound Navigation System ===")
        print("1. Test environment interactively")
        print("2. Train agents for all tasks")
        print("3. Exit")
        
        try:
            choice = int(input("Select an option (1-3): "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        if choice == 1:
            test_interface()
        elif choice == 2:
            print("Training interface would go here.")
            # In a real implementation, this would call training functions
        elif choice == 3:
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main_menu()