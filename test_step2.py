#!/usr/bin/env python
# Test script for Stage 2 implementation
import numpy as np
from core.grid_world import GridWorld, Agent
from core.sound_source import SoundSource, Wall

def test_stage2():
    print("Testing Stage 2: Physics of sound and walls")
    
    # Create a grid world
    world = GridWorld()
    
    # Place some walls with different permeabilities
    world.place_wall(5, 5, permeability=0.3)
    world.place_wall(5, 6, permeability=0.7)
    world.place_wall(5, 7, permeability=0.9)
    
    # Place a sound source
    sound_source = SoundSource(10, 10, volume=0.8, frequency=0.9)
    world.place_sound_source(sound_source)
    
    # Place the agent
    world.place_agent(12, 12)
    
    # Compute sound map
    sound_map = world.compute_sound_map()
    
    print("Sound map computed successfully!")
    print(f"Sound map shape: {sound_map.shape}")
    print(f"Sound intensity at agent position (12, 12): {sound_map[12][12]}")
    print(f"Sound intensity at sound source (10, 10): {sound_map[10][10]}")
    print(f"Sound intensity at wall (5, 5): {sound_map[5][5]}")
    
    # Test agent observation
    agent = Agent(12, 12)
    observation = agent.observe(sound_map)
    print(f"Agent observation at (12, 12): {observation}")
    
    # Test Wall class
    wall = Wall(3, 3, permeability=0.5)
    print(f"Wall at (3, 3) with permeability: {wall.permeability}")
    print(f"Is wall passable by agent: {wall.is_passable_by_agent()}")
    
    # Test SoundSource class
    source = SoundSource(15, 15, volume=0.7, frequency=0.6)
    emitted = source.emit_sound()
    print(f"Sound source at (15, 15) emitted sound: {emitted}")
    print(f"Source volume: {source.volume}, frequency: {source.frequency}")
    
    print("\nAll Stage 2 components tested successfully!")

if __name__ == "__main__":
    test_stage2()