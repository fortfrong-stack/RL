"""
Test script for STEP 1: Basic Infrastructure
This script tests the GridWorld and Agent classes created in STEP 1.
"""

from core.grid_world import GridWorld, Agent
from core.sound_source import SoundSource

def test_basic_infrastructure():
    print("Testing STEP 1: Basic Infrastructure")
    print("-" * 40)
    
    # Test GridWorld creation
    print("1. Creating GridWorld...")
    grid = GridWorld()
    print(f"   Grid size: {grid.width}x{grid.height}")
    print(f"   Initial state shape: {grid.get_state().shape}")
    
    # Test placing elements
    print("\n2. Placing walls, sound sources, and agent...")
    grid.place_wall(5, 5)
    grid.place_wall(10, 10)
    
    source = SoundSource(3, 3, volume=0.8, frequency=0.7)
    grid.place_sound_source(source)
    
    success = grid.place_agent(0, 0)
    print(f"   Agent placed successfully: {success}")
    
    # Test agent functionality
    print("\n3. Testing Agent functionality...")
    agent = Agent(0, 0)
    print(f"   Initial agent position: {agent.get_position()}")
    
    # Test movement
    new_pos = agent.move('right', grid)
    print(f"   After moving right: {new_pos}")
    
    new_pos = agent.move(0, grid)  # Move up using integer action
    print(f"   After moving up: {new_pos}")
    
    # Test boundaries
    print("\n4. Testing boundary checks...")
    print(f"   Valid position (20, 20): {grid.is_valid_position(20, 20)}")
    print(f"   Invalid position (30, 30): {grid.is_valid_position(30, 30)}")
    
    # Test sound source
    print("\n5. Testing SoundSource functionality...")
    print(f"   Sound source at ({source.x}, {source.y}) with volume {source.volume}")
    for i in range(5):
        emits = source.emit_sound()
        print(f"   Emit sound attempt {i+1}: {emits}")
    
    print("\n6. Testing GridWorld methods...")
    print(f"   Current grid state shape: {grid.get_state().shape}")
    print("   Resetting grid...")
    grid.reset()
    print(f"   Grid state after reset: {grid.get_state().shape}")
    
    print("\n✓ All STEP 1 tests passed!")

if __name__ == "__main__":
    test_basic_infrastructure()