import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
import json
import argparse

# Disable GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model configuration
with open('model_config.json', 'r') as f:
    config = json.load(f)

SAMPLE_RATE = config['sample_rate']
DURATION = config['duration']
N_MELS = config['n_mels']
N_FFT = config['n_fft']
HOP_LENGTH = config['hop_length']

def load_audio_file(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Load audio file dan konversi ke durasi yang sama"""
    try:
        # Load audio
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Jika audio lebih pendek dari durasi yang diinginkan, pad dengan zeros
        if len(audio) < sr * duration:
            audio = np.pad(audio, (0, sr * duration - len(audio)), mode='constant')
        else:
            audio = audio[:sr * duration]
            
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_mel_spectrogram(audio, sr=SAMPLE_RATE):
    """Ekstrak mel-spectrogram dari audio"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    
    # Konversi ke dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

def predict_single_file(model, file_path, class_names):
    """Prediksi untuk satu file audio"""
    # Load audio
    audio = load_audio_file(file_path)
    if audio is None:
        return None
    
    # Ekstrak mel-spectrogram
    mel_spec = extract_mel_spectrogram(audio)
    
    # Reshape untuk model
    mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
    
    # Prediksi
    predictions = model.predict(mel_spec, verbose=0)
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'class': class_names[idx],
            'confidence': float(predictions[0][idx])
        })
    
    return results

def predict_batch(model, audio_files, class_names):
    """Prediksi untuk batch file audio"""
    results = {}
    
    for audio_file in audio_files:
        print(f"Processing: {audio_file}")
        predictions = predict_single_file(model, audio_file, class_names)
        
        if predictions:
            results[audio_file] = predictions
            print(f"  Predicted: {predictions[0]['class']} (confidence: {predictions[0]['confidence']:.3f})")
        else:
            results[audio_file] = None
            print(f"  Error processing file")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Predict bird species from audio files')
    parser.add_argument('audio_path', type=str, help='Path to audio file or directory')
    parser.add_argument('--model', type=str, default='bird_sound_classifier.h5', 
                       help='Path to trained model (default: bird_sound_classifier.h5)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions (default: 0.5)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Force CPU if requested
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Running on CPU...")
    
    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Please run train_model.py first to train the model.")
        return
    
    print(f"Loading model from {args.model}...")
    model = keras.models.load_model(args.model)
    
    # Load class names
    if not os.path.exists('class_names.json'):
        print("Error: class_names.json not found!")
        print("Please run train_model.py first to train the model.")
        return
    
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # Check if input is file or directory
    if os.path.isfile(args.audio_path):
        # Single file prediction
        print(f"\nPredicting for: {args.audio_path}")
        predictions = predict_single_file(model, args.audio_path, class_names)
        
        if predictions:
            print("\nPrediction Results:")
            print("-" * 50)
            for i, pred in enumerate(predictions):
                print(f"{i+1}. {pred['class']}: {pred['confidence']:.3f}")
                if pred['confidence'] < args.threshold:
                    print(f"   (Below threshold {args.threshold})")
        else:
            print("Error processing audio file")
            
    elif os.path.isdir(args.audio_path):
        # Batch prediction for directory
        audio_files = []
        for file in os.listdir(args.audio_path):
            if file.lower().endswith(('.wav', '.mp3')):
                audio_files.append(os.path.join(args.audio_path, file))
        
        if not audio_files:
            print(f"No audio files found in {args.audio_path}")
            return
        
        print(f"\nFound {len(audio_files)} audio files")
        results = predict_batch(model, audio_files, class_names)
        
        # Save results to JSON
        output_file = 'prediction_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")
        
    else:
        print(f"Error: '{args.audio_path}' is not a valid file or directory")

if __name__ == "__main__":
    main()
