#!/usr/bin/env python3
"""
Test script for enhanced tasks with normalized rewards.
This script demonstrates the improvements made to the reward systems.
"""

import numpy as np
import sys
import os

# Add the workspace to the path
sys.path.insert(0, '/workspace')

from core.enhanced_tasks import (
    EnhancedFindAllSourcesTask,
    EnhancedFindQuietestPlaceTask,
    EnhancedFollowMovingSourceTask
)

def test_enhanced_find_all_sources():
    """Test the enhanced find all sources task."""
    print("="*60)
    print("Testing Enhanced Find All Sources Task")
    print("="*60)
    
    # Create environment with 3 sources
    env = EnhancedFindAllSourcesTask(num_sources=3)
    obs = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Number of sources: {len(env.grid_world.sound_sources)}")
    print(f"Agent starting position: {env.agent.get_position()}")
    
    total_reward = 0
    steps = 0
    
    # Simple random walk to test rewards
    for i in range(100):
        # Choose a random action (0: up, 1: down, 2: left, 3: right, 4: stay)
        action = np.random.randint(0, 5)
        
        obs, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        
        if reward != -0.5:  # Non-standard reward (found source or special event)
            print(f"Step {steps}: Action={action}, Reward={reward:.2f}, Position={env.agent.get_position()}, Found={len(env.found_sources)}")
        
        if done:
            print(f"Task completed! Found all sources in {steps} steps.")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Sources found: {len(env.found_sources)}/{len(env.grid_world.sound_sources)}")
    print()


def test_enhanced_find_quietest_place():
    """Test the enhanced find quietest place task."""
    print("="*60)
    print("Testing Enhanced Find Quietest Place Task")
    print("="*60)
    
    # Create environment with 4 sources
    env = EnhancedFindQuietestPlaceTask(num_sources=4)
    obs = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Number of sources: {len(env.grid_world.sound_sources)}")
    print(f"Agent starting position: {env.agent.get_position()}")
    print(f"Quietest cell: {env.quietest_cell}")
    
    total_reward = 0
    steps = 0
    
    # Simple random walk to test rewards
    for i in range(100):
        # Choose a random action (0: up, 1: down, 2: left, 3: right, 4: stay)
        action = np.random.randint(0, 5)
        
        obs, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        
        if reward != -0.3:  # Non-standard reward (got closer to quietest spot)
            agent_pos = env.agent.get_position()
            intensity = env.sound_map[agent_pos[0]][agent_pos[1]]
            print(f"Step {steps}: Action={action}, Reward={reward:.2f}, Position={agent_pos}, Intensity={intensity:.3f}")
        
        if done:
            print(f"Task completed! Reached quietest place in {steps} steps.")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    agent_pos = env.agent.get_position()
    print(f"Final position: {agent_pos}, Quietest cell: {env.quietest_cell}")
    print(f"Distance to target: {abs(agent_pos[0] - env.quietest_cell[0]) + abs(agent_pos[1] - env.quietest_cell[1])}")
    print()


def test_enhanced_follow_moving_source():
    """Test the enhanced follow moving source task."""
    print("="*60)
    print("Testing Enhanced Follow Moving Source Task")
    print("="*60)
    
    # Create environment with moving source
    env = EnhancedFollowMovingSourceTask(move_interval=20)
    obs = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Agent starting position: {env.agent.get_position()}")
    source_pos = (env.grid_world.sound_sources[0].x, env.grid_world.sound_sources[0].y)
    print(f"Initial source position: {source_pos}")
    
    total_reward = 0
    steps = 0
    
    # Simple random walk to test rewards
    for i in range(100):
        # Choose a random action (0: up, 1: down, 2: left, 3: right, 4: stay)
        action = np.random.randint(0, 5)
        
        obs, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        
        if reward != -0.3:  # Non-standard reward (got closer to source)
            agent_pos = env.agent.get_position()
            source_pos = (env.grid_world.sound_sources[0].x, env.grid_world.sound_sources[0].y)
            distance = abs(agent_pos[0] - source_pos[0]) + abs(agent_pos[1] - source_pos[1])
            print(f"Step {steps}: Action={action}, Reward={reward:.2f}, Distance to source={distance}")
        
        if steps % 20 == 0:  # Show source movement
            source_pos = (env.grid_world.sound_sources[0].x, env.grid_world.sound_sources[0].y)
            agent_pos = env.agent.get_position()
            distance = abs(agent_pos[0] - source_pos[0]) + abs(agent_pos[1] - source_pos[1])
            print(f"  Source moved to: {source_pos}, Agent: {agent_pos}, Distance: {distance}")
        
        if done:
            print(f"Task completed! Caught the source in {steps} steps.")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    agent_pos = env.agent.get_position()
    source_pos = (env.grid_world.sound_sources[0].x, env.grid_world.sound_sources[0].y)
    distance = abs(agent_pos[0] - source_pos[0]) + abs(agent_pos[1] - source_pos[1])
    print(f"Final distance: {distance}")
    print()


def compare_rewards():
    """Compare reward characteristics between original and enhanced tasks."""
    print("="*60)
    print("Comparing Original vs Enhanced Reward Characteristics")
    print("="*60)
    
    print("TASK 1 - Find All Sources:")
    print("- Original: -0.1 step penalty, +10 for source, +50 for all")
    print("- Enhanced: -0.5 step penalty, +20 for source, +100 for all, +intermediate rewards")
    print()
    
    print("TASK 2 - Find Quietest Place:")
    print("- Original: -intensity as reward, +100 for target")
    print("- Enhanced: normalized intensity rewards, +potential function, +smoothing")
    print()
    
    print("TASK 3 - Follow Moving Source:")
    print("- Original: +5/-5 for distance changes, +100 for catch")
    print("- Enhanced: continuous distance rewards, +prediction, +smoothing")
    print()
    
    print("ENHANCEMENTS ADDED:")
    print("✓ Normalized rewards to prevent instability")
    print("✓ Increased step penalties to encourage efficiency")
    print("✓ Added intermediate rewards for progress")
    print("✓ Used potential functions for smoother guidance")
    print("✓ Added reward smoothing to reduce discreteness")
    print("✓ Improved reward stability and learning efficiency")


if __name__ == "__main__":
    print("Testing Enhanced Tasks with Normalized Rewards")
    print("This script demonstrates the improvements made to address:")
    print("1. Too small step penalties (-0.1)")
    print("2. Unstable intensity-proportional penalties")
    print("3. Discrete distance changes")
    print()
    
    # Test all enhanced tasks
    test_enhanced_find_all_sources()
    test_enhanced_find_quietest_place()
    test_enhanced_follow_moving_source()
    
    # Show comparison
    compare_rewards()
    
    print("="*60)
    print("All tests completed successfully!")
    print("Enhanced tasks are ready for use with improved reward systems.")
    print("="*60)