import soundfile as sf
import librosa
import numpy as np
import os


def preprocess_audio(audio_path, sr=22050, duration=3):
    """
    Preprocessing file audio WAV
    Parameters:
        audio_path: path file audio
        sr: sample rate (default 22050 Hz)
        duration: durasi potong audio dalam detik
    Returns:
        normalized_audio: audio yang sudah dinormalisasi
    """
    try:
        # Load audio file dengan parameter tambahan
        audio, sr = librosa.load(audio_path, sr=sr, res_type='kaiser_fast', duration=duration)
        # Jika loading berhasil tapi audio kosong
        if len(audio) == 0:
            print(f"Warning: Empty audio file - {audio_path}")
            return None

        # Normalisasi amplitudo (-1 to 1)
        normalized_audio = librosa.util.normalize(audio)

        # Potong audio ke durasi yang diinginkan
        if len(normalized_audio) > sr * duration:
            normalized_audio = normalized_audio[:sr * duration]
        else:
            # Pad dengan silence jika audio terlalu pendek
            normalized_audio = librosa.util.fix_length(
                normalized_audio,
                size=sr * duration
            )

        return normalized_audio
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def extract_features(audio):
    """
    Ekstraksi fitur dari audio yang sudah dipreprocess
    Parameters:
        audio: array audio yang sudah dinormalisasi
    Returns:
        features: dictionary berisi fitur-fitur audio
    """
    # Extract MFCC
    mfccs = librosa.feature.mfcc(y=audio, n_mfcc=13)

    features = {
        f'mfcc_{i}': mfcc for i, mfcc in enumerate(np.mean(mfccs, axis=1))
    }

    # Tambahkan fitur lainnya
    features.update({
        'zcr': np.mean(librosa.feature.zero_crossing_rate(audio)),
        'centroid': np.mean(librosa.feature.spectral_centroid(y=audio)),
        'bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=audio)),
        'rms': np.mean(librosa.feature.rms(y=audio))
    })

    return features
