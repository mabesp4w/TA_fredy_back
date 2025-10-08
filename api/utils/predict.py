import os
import numpy as np
import librosa
import tensorflow as tf
import json
import logging
from django.conf import settings

# Use Keras 3.x (standalone) - compatible with YAMNet model
import keras

from api.utils.file_storage import audio_storage
from crud.serializers import BirdDetailSerializer
from crud.models import Bird
# from api.utils.mlburung_predict import predict_with_mlburung_model  # Removed - using new model

# Disable GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)


def load_model_config():
    """Load model configuration"""
    config_path = os.path.join(settings.BASE_DIR, 'models/model_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_class_names():
    """Load class names"""
    class_names_path = os.path.join(settings.BASE_DIR, 'models/class_names.json')
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file not found at: {class_names_path}")
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def load_model(model_path=None):
    """Load the trained YAMNet model with Keras 3.x"""
    if model_path is None:
        model_path = os.path.join(settings.BASE_DIR, 'models/bird_sound_classifier.h5')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        # Load model with Keras 3.x
        logger.info(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully with Keras 3.x")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise Exception(f"Failed to load model: {e}")


def load_audio_file(file_path, sr=22050, duration=5):
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
        logger.error(f"Error loading {file_path}: {e}")
        return None


def extract_mel_spectrogram(audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """Ekstrak mel-spectrogram dari audio"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Konversi ke dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def save_audio_tempfile(audio_file):
    """
    Menyimpan file audio menggunakan custom storage
    """
    try:
        logger.info(f"Saving audio file: {getattr(audio_file, 'name', 'unknown')}")

        # Gunakan custom storage
        temp_path = audio_storage.save_temp_audio(audio_file)

        if temp_path:
            logger.info(f"Audio saved successfully to: {temp_path}")
            return temp_path
        else:
            logger.error("Failed to save audio file using custom storage")
            return None

    except Exception as e:
        logger.error(f"Error in save_audio_tempfile: {e}")
        return None


def predict_single_audio(audio_file, model=None, config=None, class_names=None, use_mlburung_fallback=False):
    """
    Predict bird species from audio file using TensorFlow model
    """
    audio_path = None
    try:
        logger.info(f"Starting prediction for audio file: {getattr(audio_file, 'name', 'unknown')}")

        # Validasi input
        if not audio_file:
            logger.error("No audio file provided")
            return {"error": "No audio file provided"}

        # MLBurung model fallback disabled - using new YAMNet model
        # if use_mlburung_fallback:
        #     try:
        #         logger.info("Trying MLBurung model first...")
        #         mlburung_result = predict_with_mlburung_model(audio_file)
        #         
        #         if 'error' not in mlburung_result:
        #             # Get bird data from database
        #             try:
        #                 bird = Bird.objects.filter(scientific_nm=mlburung_result['scientific_nm']).first()
        #                 if bird:
        #                     bird_data = BirdDetailSerializer(bird).data
        #                     mlburung_result['bird_data'] = bird_data
        #                 else:
        #                     mlburung_result['bird_data'] = None
        #             except Exception as e:
        #                 logger.warning(f"Failed to get bird data: {e}")
        #                 mlburung_result['bird_data'] = None
        #             
        #             logger.info(f"MLBurung prediction successful: {mlburung_result['scientific_nm']}")
        #             return mlburung_result
        #         else:
        #             logger.warning(f"MLBurung model failed: {mlburung_result['error']}")
        #     except Exception as e:
        #         logger.warning(f"MLBurung model error: {e}")

        # Using YAMNet TensorFlow model
        logger.info("Using YAMNet TensorFlow model...")

        # Load model config jika belum ada
        if config is None:
            try:
                config = load_model_config()
                logger.info("Model config loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model config: {e}")
                return {"error": f"Failed to load model config: {str(e)}"}

        # Load class names jika belum ada
        if class_names is None:
            try:
                class_names = load_class_names()
                logger.info("Class names loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load class names: {e}")
                return {"error": f"Failed to load class names: {str(e)}"}

        # Load model jika belum ada
        if model is None:
            try:
                model = load_model()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return {"error": f"Failed to load model: {str(e)}"}

        # Simpan file audio yang diupload ke file sementara
        audio_path = save_audio_tempfile(audio_file)
        if not audio_path:
            logger.error("Failed to save audio file to temporary location")
            return {"error": "Failed to save audio file"}

        logger.info(f"Audio saved to temporary path: {audio_path}")

        # Load audio file
        try:
            audio = load_audio_file(
                audio_path, 
                sr=config['sample_rate'], 
                duration=config['duration']
            )
            if audio is None:
                logger.error("Failed to load audio file")
                return {"error": "Failed to load audio file"}
            logger.info("Audio loaded successfully")
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return {"error": f"Failed to load audio: {str(e)}"}

        # Extract mel-spectrogram
        try:
            mel_spec = extract_mel_spectrogram(
                audio,
                sr=config['sample_rate'],
                n_mels=config['n_mels'],
                n_fft=config['n_fft'],
                hop_length=config['hop_length']
            )
            logger.info("Mel-spectrogram extraction completed successfully")
        except Exception as e:
            logger.error(f"Error during mel-spectrogram extraction: {e}")
            return {"error": f"Failed to extract mel-spectrogram: {str(e)}"}

        # Reshape untuk model
        try:
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
            logger.info(f"Mel-spectrogram shape: {mel_spec.shape}")
        except Exception as e:
            logger.error(f"Error reshaping mel-spectrogram: {e}")
            return {"error": f"Failed to reshape mel-spectrogram: {str(e)}"}

        # Predict
        try:
            predictions = model.predict(mel_spec, verbose=0)
            
            # Get top prediction
            top_index = np.argmax(predictions[0])
            prediction = class_names[top_index]
            confidence = float(predictions[0][top_index])

            logger.info(f"TensorFlow prediction: {prediction}, Confidence: {confidence}")
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            return {"error": f"Failed to make prediction: {str(e)}"}

        # Ambil bird berdasarkan scientific_nm yang diprediksi
        try:
            bird = Bird.objects.filter(scientific_nm=prediction).first()
            if not bird:
                logger.warning(f"No bird found for scientific name: {prediction}")
                return {
                    'scientific_nm': prediction,
                    'confidence': confidence,
                    'bird_data': None,
                    'method': 'yamnet_model'
                }

            # Serialize the bird data using the BirdDetailSerializer
            bird_data = BirdDetailSerializer(bird).data

            logger.info(f"Successfully completed YAMNet prediction for {prediction}")
            return {
                'scientific_nm': prediction,
                'confidence': confidence,
                'bird_data': bird_data,
                'method': 'yamnet_model'
            }

        except Exception as e:
            logger.error(f"Error retrieving bird data: {e}")
            return {"error": f"Failed to retrieve bird data: {str(e)}"}

    except Exception as e:
        logger.error(f"Unexpected error in predict_single_audio: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

    finally:
        # Cleanup temporary file
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
                logger.info(f"Cleaned up temporary file: {audio_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {audio_path}: {cleanup_error}")


def predict_from_file_path(file_path, model=None, config=None, class_names=None, use_cpu=False):
    """
    Predict bird species from audio file path (similar to MLBurung predict.py)
    """
    try:
        logger.info(f"Starting prediction for audio file: {file_path}")

        # Force CPU if requested
        if use_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            logger.info("Running on CPU...")

        # Load model config jika belum ada
        if config is None:
            try:
                config = load_model_config()
                logger.info("Model config loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model config: {e}")
                return {"error": f"Failed to load model config: {str(e)}"}

        # Load class names jika belum ada
        if class_names is None:
            try:
                class_names = load_class_names()
                logger.info("Class names loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load class names: {e}")
                return {"error": f"Failed to load class names: {str(e)}"}

        # Load model jika belum ada
        if model is None:
            try:
                model = load_model()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return {"error": f"Failed to load model: {str(e)}"}

        # Load audio file
        try:
            audio = load_audio_file(
                file_path, 
                sr=config['sample_rate'], 
                duration=config['duration']
            )
            if audio is None:
                logger.error("Failed to load audio file")
                return {"error": "Failed to load audio file"}
            logger.info("Audio loaded successfully")
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return {"error": f"Failed to load audio: {str(e)}"}

        # Extract mel-spectrogram
        try:
            mel_spec = extract_mel_spectrogram(
                audio,
                sr=config['sample_rate'],
                n_mels=config['n_mels'],
                n_fft=config['n_fft'],
                hop_length=config['hop_length']
            )
            logger.info("Mel-spectrogram extraction completed successfully")
        except Exception as e:
            logger.error(f"Error during mel-spectrogram extraction: {e}")
            return {"error": f"Failed to extract mel-spectrogram: {str(e)}"}

        # Reshape untuk model
        try:
            mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
            logger.info(f"Mel-spectrogram shape: {mel_spec.shape}")
        except Exception as e:
            logger.error(f"Error reshaping mel-spectrogram: {e}")
            return {"error": f"Failed to reshape mel-spectrogram: {str(e)}"}

        # Predict
        try:
            predictions = model.predict(mel_spec, verbose=0)
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'class': class_names[idx],
                    'confidence': float(predictions[0][idx])
                })

            logger.info(f"Top prediction: {results[0]['class']} (confidence: {results[0]['confidence']:.3f})")
            return results

        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            return {"error": f"Failed to make prediction: {str(e)}"}

    except Exception as e:
        logger.error(f"Unexpected error in predict_from_file_path: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


def predict_batch_files(audio_files, model=None, config=None, class_names=None, use_cpu=False):
    """
    Predict bird species for batch of audio files
    """
    results = {}
    
    for audio_file in audio_files:
        logger.info(f"Processing: {audio_file}")
        predictions = predict_from_file_path(audio_file, model, config, class_names, use_cpu)
        
        if isinstance(predictions, list):
            results[audio_file] = predictions
            logger.info(f"  Predicted: {predictions[0]['class']} (confidence: {predictions[0]['confidence']:.3f})")
        else:
            results[audio_file] = None
            logger.error(f"  Error processing file: {predictions.get('error', 'Unknown error')}")
    
    return results


def main():
    """
    Main function for command line interface (similar to MLBurung predict.py)
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict bird species from audio files')
    parser.add_argument('audio_path', type=str, help='Path to audio file or directory')
    parser.add_argument('--model', type=str, default='models/bird_sound_classifier.h5', 
                       help='Path to trained model (default: models/bird_sound_classifier.h5)')
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
        print("Please make sure the model file exists.")
        return
    
    print(f"Loading model from {args.model}...")
    model = keras.models.load_model(args.model)
    
    # Load config and class names
    try:
        config = load_model_config()
        class_names = load_class_names()
    except Exception as e:
        print(f"Error loading config or class names: {e}")
        return
    
    # Check if input is file or directory
    if os.path.isfile(args.audio_path):
        # Single file prediction
        print(f"\nPredicting for: {args.audio_path}")
        predictions = predict_from_file_path(args.audio_path, model, config, class_names, args.cpu)
        
        if isinstance(predictions, list):
            print("\nPrediction Results:")
            print("-" * 50)
            for i, pred in enumerate(predictions):
                print(f"{i+1}. {pred['class']}: {pred['confidence']:.3f}")
                if pred['confidence'] < args.threshold:
                    print(f"   (Below threshold {args.threshold})")
        else:
            print(f"Error processing audio file: {predictions.get('error', 'Unknown error')}")
            
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
        results = predict_batch_files(audio_files, model, config, class_names, args.cpu)
        
        # Save results to JSON
        output_file = 'prediction_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")
        
    else:
        print(f"Error: '{args.audio_path}' is not a valid file or directory")


if __name__ == "__main__":
    main()