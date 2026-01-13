import random


class SoundSource:
    """
    Class representing a sound source in the grid world.
    """
    
    def __init__(self, x, y, volume=0.5, frequency=0.5):
        self.x = x
        self.y = y
        self.volume = volume  # Determines the radius of sound propagation (0.1-1.0)
        self.frequency = frequency  # Probability of emitting sound per step (0.1-1.0)
        
    def emit_sound(self):
        """
        Generate sound signal with given probability based on frequency.
        Returns True if sound is emitted, False otherwise.
        """
        return random.random() < self.frequency