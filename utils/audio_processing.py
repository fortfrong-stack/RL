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
    # that mimics the general process of MFCC extraction
    
    # Calculate spectrogram using STFT (simplified)
    frame_length = min(len(audio_signal), 1024)
    hop_length = frame_length // 4
    
    # Split signal into frames
    n_frames = max(1, (len(audio_signal) - frame_length) // hop_length + 1)
    frames = []
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, len(audio_signal))
        frame = audio_signal[start:end]
        # Apply window function (Hamming window approximation)
        windowed_frame = frame * np.hamming(len(frame))
        frames.append(windowed_frame)
    
    if len(frames) == 0:
        # If the signal is too short, just use the entire signal
        windowed_signal = audio_signal * np.hamming(len(audio_signal))
        frames = [windowed_signal]
    
    # Calculate power spectrum for each frame
    power_spectra = []
    for frame in frames:
        fft = np.fft.rfft(frame)
        power_spectrum = np.abs(fft) ** 2
        power_spectra.append(power_spectrum)
    
    # Average power spectrum across all frames
    avg_power_spectrum = np.mean(power_spectra, axis=0) if power_spectra else np.zeros(frame_length//2 + 1)
    
    # Apply mel filter bank (simplified approach)
    n_filters = max(n_mfcc, 26)  # Common number of mel filters
    mel_filters = []
    
    # Create simplified triangular mel filters
    n_fft = len(avg_power_spectrum)
    low_freq = 0
    high_freq = 4000  # Assume 4kHz as upper limit for simplicity
    # Convert Hz to mel
    low_mel = 2595 * np.log10(1 + low_freq / 700.0)
    high_mel = 2595 * np.log10(1 + high_freq / 700.0)
    
    # Create equally spaced mel frequencies
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = 700 * (10**(mel_points / 2595.0) - 1)
    
    # Convert back to bin indices
    bin_indices = np.floor((n_fft + 1) * hz_points / (high_freq * 2)).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft)
    
    # Create triangular filters
    for i in range(1, n_filters + 1):
        filter_bank = np.zeros(n_fft)
        
        left = bin_indices[i-1]
        center = bin_indices[i]
        right = bin_indices[i+1]
        
        if left == center and center == right:
            continue
            
        # Triangular filter
        for j in range(left, center):
            if j < len(filter_bank):
                filter_bank[j] = (j - left) / (center - left)
        for j in range(center, right):
            if j < len(filter_bank):
                filter_bank[j] = (right - j) / (right - center)
        
        # Apply filter to power spectrum
        filtered_energy = np.sum(avg_power_spectrum * filter_bank)
        mel_filters.append(filtered_energy)
    
    if len(mel_filters) == 0:
        # Fallback to basic features if mel filtering fails
        mean_val = np.mean(audio_signal)
        std_val = np.std(audio_signal)
        max_val = np.max(audio_signal)
        min_val = np.min(audio_signal)
        median_val = np.median(audio_signal)
        rms = np.sqrt(np.mean(audio_signal ** 2))
        zcr = np.mean(np.abs(np.diff(np.sign(audio_signal))))
        
        features = np.array([mean_val, std_val, max_val, min_val, median_val, rms, zcr])
    else:
        # Take log of mel energies
        log_mel_energies = np.log(np.maximum(mel_filters, 1e-10))  # Avoid log(0)
        
        # Apply DCT to get MFCCs (simplified)
        # Using scipy.fftpack.dct would be ideal, but we'll approximate
        try:
            # Use a simple approach to get cepstral coefficients
            mfcc_coeffs = np.zeros(n_mfcc)
            for i in range(n_mfcc):
                if i < len(log_mel_energies):
                    # Simple weighted sum to simulate DCT effect
                    weights = np.cos(i * np.pi * np.arange(len(log_mel_energies)) / len(log_mel_energies))
                    mfcc_coeffs[i] = np.sum(log_mel_energies * weights)
        except:
            # If DCT simulation fails, use simpler approach
            mfcc_coeffs = np.array(log_mel_energies[:n_mfcc])
        
        features = mfcc_coeffs
    
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
    try:
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
        
        # Normalize features to prevent large values from dominating
        combined_features = np.clip(combined_features, -100, 100)  # Prevent extreme values
        
        return combined_features
    except Exception as e:
        # Fallback to basic features if anything goes wrong
        print(f"Warning: Error in audio feature extraction: {e}")
        # Return a default feature vector
        return np.zeros(20)  # Default size based on typical MFCC + spectral features + intensity