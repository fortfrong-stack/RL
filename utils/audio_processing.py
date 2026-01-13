"""
Audio processing module for extracting features from audio signals.
Implements MFCC extraction and other audio feature extraction methods.
"""

import numpy as np


def generate_audio_signal(intensity, frequency_content, duration=0.1, sample_rate=44100):
    """
    Generate synthetic audio signal based on intensity and frequency content.
    
    Args:
        intensity: Strength of sound in the cell
        frequency_content: Characteristic frequency of the source
        duration: Duration of the signal in seconds
        sample_rate: Sample rate for audio generation
    
    Returns:
        Generated audio signal as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate signal with specified frequency
    base_freq = 440 * frequency_content  # Base frequency adjusted by content
    signal = intensity * np.sin(2 * np.pi * base_freq * t)
    
    # Add noise and processing
    noise = np.random.normal(0, 0.01, len(t))
    return signal + noise


def extract_mfcc_features(audio_signal, n_mfcc=13):
    """
    Extract MFCC (Mel-frequency cepstral coefficients) features from audio signal.
    Simplified implementation since librosa might not be available.
    
    Args:
        audio_signal: Input audio signal as numpy array
        n_mfcc: Number of MFCC coefficients to extract
    
    Returns:
        MFCC features as numpy array
    """
    # Since librosa might not be available, implementing a simplified version
    # This is a placeholder implementation that extracts basic statistical features
    # that would mimic what MFCC represents
    
    # Split signal into frames
    frame_length = min(len(audio_signal), 1024)
    hop_length = frame_length // 4
    
    # Simple approach: calculate basic statistical features that represent MFCC concepts
    mean_val = np.mean(audio_signal)
    std_val = np.std(audio_signal)
    max_val = np.max(audio_signal)
    min_val = np.min(audio_signal)
    median_val = np.median(audio_signal)
    rms = np.sqrt(np.mean(audio_signal ** 2))
    zcr = np.mean(np.abs(np.diff(np.sign(audio_signal))))
    
    # Return a vector of basic features (not true MFCC but representative features)
    features = np.array([mean_val, std_val, max_val, min_val, median_val, rms, zcr])
    
    # Pad or truncate to desired length
    if len(features) >= n_mfcc:
        return features[:n_mfcc]
    else:
        # Pad with zeros if needed
        padded = np.zeros(n_mfcc)
        padded[:len(features)] = features
        return padded


def extract_spectral_features(audio_signal):
    """
    Extract spectral features from audio signal.
    Simplified implementation without librosa.
    
    Args:
        audio_signal: Input audio signal as numpy array
    
    Returns:
        Spectral features as numpy array
    """
    # Calculate FFT
    fft = np.fft.fft(audio_signal)
    magnitude_spectrum = np.abs(fft[:len(fft)//2])  # Only positive frequencies
    
    if len(magnitude_spectrum) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    # Calculate spectral centroid (center of mass of spectrum)
    freq_indices = np.arange(len(magnitude_spectrum))
    if np.sum(magnitude_spectrum) == 0:
        spectral_centroid = 0
    else:
        spectral_centroid = np.sum(freq_indices * magnitude_spectrum) / np.sum(magnitude_spectrum)
    
    # Calculate spectral rolloff (frequency below which 85% of energy lies)
    cumulative_energy = np.cumsum(magnitude_spectrum)
    total_energy = cumulative_energy[-1]
    rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
    spectral_rolloff = rolloff_idx[0] if len(rolloff_idx) > 0 else len(magnitude_spectrum) - 1
    
    # Calculate zero crossing rate
    zcr = np.mean(np.abs(np.diff(np.sign(audio_signal))))
    
    # Calculate spectral bandwidth (variance of spectrum around centroid)
    if np.sum(magnitude_spectrum) == 0:
        spectral_bandwidth = 0
    else:
        spectral_bandwidth = np.sqrt(
            np.sum(((freq_indices - spectral_centroid) ** 2) * magnitude_spectrum) / 
            np.sum(magnitude_spectrum)
        )
    
    return np.array([spectral_centroid, spectral_rolloff, zcr, spectral_bandwidth])


def get_audio_observation_features(intensity, frequency_content, sample_rate=44100):
    """
    Get comprehensive audio observation features for the agent.
    
    Args:
        intensity: Sound intensity at agent's position
        frequency_content: Frequency characteristic of the source
        sample_rate: Sample rate for audio processing
    
    Returns:
        Combined feature vector as numpy array
    """
    # Generate synthetic audio signal
    audio_signal = generate_audio_signal(intensity, frequency_content, sample_rate=sample_rate)
    
    # Extract MFCC features
    mfcc_features = extract_mfcc_features(audio_signal)
    
    # Extract spectral features
    spectral_features = extract_spectral_features(audio_signal)
    
    # Combine features into a single observation vector
    combined_features = np.concatenate([
        mfcc_features,
        spectral_features,
        [intensity]  # Include raw intensity as well
    ])
    
    return combined_features