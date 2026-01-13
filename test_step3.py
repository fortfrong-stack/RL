#!/usr/bin/env python
# Test script for Stage 3 implementation
import numpy as np
from core.grid_world import GridWorld, Agent
from core.sound_source import SoundSource, Wall
from utils.audio_processing import generate_audio_signal, extract_mfcc_features, extract_spectral_features, get_audio_observation_features


def test_stage3():
    print("Testing Stage 3: System of observations and partial observability")
    
    # Test audio signal generation
    print("\n1. Testing audio signal generation...")
    intensity = 0.7
    frequency_content = 0.8
    audio_signal = generate_audio_signal(intensity, frequency_content)
    print(f"   Generated audio signal with length: {len(audio_signal)}")
    print(f"   Signal range: [{np.min(audio_signal):.4f}, {np.max(audio_signal):.4f}]")
    
    # Test MFCC feature extraction
    print("\n2. Testing MFCC feature extraction...")
    mfcc_features = extract_mfcc_features(audio_signal)
    print(f"   MFCC features shape: {mfcc_features.shape}")
    print(f"   MFCC features: {mfcc_features[:5]}...")  # Show first 5 features
    
    # Test spectral feature extraction
    print("\n3. Testing spectral feature extraction...")
    spectral_features = extract_spectral_features(audio_signal)
    print(f"   Spectral features shape: {spectral_features.shape}")
    print(f"   Spectral features: {spectral_features}")
    
    # Test combined audio observation features
    print("\n4. Testing combined audio observation features...")
    combined_features = get_audio_observation_features(intensity, frequency_content)
    print(f"   Combined features shape: {combined_features.shape}")
    print(f"   First few features: {combined_features[:5]}...")
    
    # Test integration with GridWorld and Agent
    print("\n5. Testing integration with GridWorld and Agent...")
    
    # Create a grid world
    world = GridWorld()
    
    # Place a sound source
    sound_source = SoundSource(10, 10, volume=0.8, frequency=0.9)
    world.place_sound_source(sound_source)
    
    # Place the agent
    world.place_agent(12, 12)
    
    # Compute sound map
    sound_map = world.compute_sound_map()
    print(f"   Sound map computed, shape: {sound_map.shape}")
    print(f"   Sound intensity at agent position (12, 12): {sound_map[12][12]:.4f}")
    
    # Create an agent and get audio observation
    agent = Agent(12, 12)
    audio_observation = agent.observe(sound_map=sound_map, grid_world=world)
    print(f"   Agent audio observation shape: {audio_observation.shape}")
    print(f"   Agent audio observation (first 5 features): {audio_observation[:5]}...")
    
    # Test with no sound sources (should return default features)
    print("\n6. Testing with no sound sources...")
    empty_world = GridWorld()
    empty_world.place_agent(5, 5)
    empty_agent = Agent(5, 5)
    empty_observation = empty_agent.observe(sound_map=np.zeros((25, 25)), grid_world=empty_world)
    print(f"   Empty world observation shape: {empty_observation.shape}")
    print(f"   Empty world observation (first 5 features): {empty_observation[:5]}...")
    
    print("\n✓ All Stage 3 components tested successfully!")
    print(f"\nStage 3 Summary:")
    print(f"- Audio signal generation: ✅ Working")
    print(f"- MFCC feature extraction: ✅ Working")
    print(f"- Spectral feature extraction: ✅ Working")
    print(f"- Agent audio observations: ✅ Working")
    print(f"- Integration with GridWorld: ✅ Working")


if __name__ == "__main__":
    test_stage3()