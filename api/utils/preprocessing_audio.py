# utils/preporcessing_audio.py
import tempfile

import resampy
import soundfile as sf
import librosa
import numpy as np
import os
import pandas as pd

# Setup logger
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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
    logger.debug(f"librosa version: {librosa.__version__}")
    logger.debug(f"soundfile version: {sf.__version__}")
    try:
        # Cek apakah server dapat menulis ke folder sementara
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_audio_path = tmp_file.name
            if not os.access(tmp_audio_path, os.W_OK):
                logger.error(f"Error: Permission denied for writing to temporary file: {tmp_audio_path}")
                return {"error": "Permission denied for writing temporary files"}
        logger.info(f"Processing audio: {audio_path}")
        # Cek apakah file bisa dibaca
        if not os.access(audio_path, os.R_OK):
            logger.error(f"Error: Permission denied for reading {audio_path}")
            return None
       # Load audio file dengan parameter tambahan
        audio, sr = librosa.load(audio_path, sr=sr, res_type='kaiser_fast', duration=duration)
        # Jika loading berhasil tapi audio kosong
        if len(audio) == 0:
            logger.error(f"Warning: Empty audio file - {audio_path}")
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
        
        # Hapus silent parts
        non_silent = librosa.effects.split(
            normalized_audio,
            top_db=20,
            frame_length=2048,
            hop_length=512
        )
        
        return normalized_audio
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {str(e)}")
        return None
    
def validate_audio(audio_path):
    """
    Validasi dan perbaiki file audio jika diperlukan
    """
    try:
        # Baca audio dengan soundfile
        audio, sr = sf.read(audio_path)
        
        # Tulis ulang file dengan format yang benar
        output_path = audio_path.replace('.wav', '_fixed.wav')
        sf.write(output_path, audio, sr, format='WAV')
        
        return output_path
    except Exception as e:
        logger.error(f"Cannot fix {audio_path}: {str(e)}")
        return None
    
def check_audio_format(audio_path):
    """
    Periksa format file audio
    """
    try:
        info = sf.info(audio_path)
        logger.info(f"File: {audio_path}")
        logger.info(f"Format: {info.format}")
        logger.info(f"Channels: {info.channels}")
        logger.info(f"Sample rate: {info.samplerate}")
        logger.info(f"Duration: {info.duration} seconds")
        return True
    except Exception as e:
        logger.error(f"Invalid audio file {audio_path}: {str(e)}")
        return False

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

def process_dataset(base_path):
    """
    Proses seluruh dataset
    Parameters:
        base_path: path folder dataset
    Returns:
        features_df: DataFrame berisi fitur-fitur seluruh audio
    """
    features_list = []
    labels = []
    
    # Iterasi setiap folder spesies
    for species in os.listdir(base_path):
        species_path = os.path.join(base_path, species)
        
        if os.path.isdir(species_path):
            # Iterasi setiap file audio dalam folder spesies
            for audio_file in os.listdir(species_path):
                if audio_file.endswith('.wav'):
                    audio_path = os.path.join(species_path, audio_file)
                    # Periksa format
                    if not check_audio_format(audio_path):
                        continue
                    # Validasi dan perbaiki jika perlu
                    # fixed_path = validate_audio(audio_path)
                    # if fixed_path is None:
                    #     continue
                    # Gunakan file yang sudah diperbaiki
                    processed_audio = preprocess_audio(audio_path)
                    if processed_audio is None:
                        continue
                    
                    features = extract_features(processed_audio)
                    features_list.append(features)
                    labels.append(species)
    
    return pd.DataFrame(features_list), labels

