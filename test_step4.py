"""
Test script for Stage 4 Task 1: Finding all sound sources
"""

import numpy as np
from core.tasks import FindAllSourcesTask
from core.sound_source import SoundSource, Wall


def test_find_all_sources_task():
    """Test the FindAllSourcesTask implementation."""
    print("Testing FindAllSourcesTask...")
    
    # Create a task environment for finding all sources
    env = FindAllSourcesTask(width=10, height=10, num_sources=3)
    
    # Manually place some walls to make it interesting
    env.grid_world.place_wall(2, 2, 0.5)
    env.grid_world.place_wall(3, 3, 0.3)
    
    # Test the environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Number of sound sources: {len(env.grid_world.sound_sources)}")
    print(f"Initial found sources: {len(env.found_sources)}")
    
    # Simulate a few steps to see if rewards work
    total_reward = 0
    for step in range(20):
        # Simple random walk to try to find sources
        action = np.random.choice(list(env.agent.actions.keys()))
        obs, reward, done = env.step(action)
        total_reward += reward
        
        agent_pos = env.agent.get_position()
        print(f"Step {step}: Action={env.agent.actions[action]}, Position={agent_pos}, Reward={reward:.2f}, Found={len(env.found_sources)}/{len(env.grid_world.sound_sources)}")
        
        if done:
            print(f"Environment terminated at step {step}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Found {len(env.found_sources)} out of {len(env.grid_world.sound_sources)} sources")
    print("Test completed!")


def test_task_creation():
    """Test that all task types can be created."""
    print("\nTesting task creation...")
    
    from core.tasks import create_task_environment
    
    # Test all task types
    task1 = create_task_environment(1, width=5, height=5, num_sources=2)
    print(f"Task 1 (find all sources) created: {type(task1).__name__}")
    
    task2 = create_task_environment(2, width=5, height=5, num_sources=2)
    print(f"Task 2 (find quietest place) created: {type(task2).__name__}")
    
    task3 = create_task_environment(3, width=5, height=5, move_interval=5)
    print(f"Task 3 (follow moving source) created: {type(task3).__name__}")
    
    print("All task types created successfully!")


if __name__ == "__main__":
    test_task_creation()
    test_find_all_sources_task()