import os
import numpy as np
import librosa
import json
import logging
from django.conf import settings
import traceback  # Untuk debug

# Ganti TensorFlow dengan tflite-runtime (ringan)
from tflite_runtime.interpreter import Interpreter

from api.utils.file_storage import audio_storage
from crud.serializers import BirdDetailSerializer
from crud.models import Bird

# Disable warnings (tidak ada TF lagi)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU untuk stability

logger = logging.getLogger(__name__)


def load_model_config():
    """Load model configuration (updated for bird model)"""
    config_path = os.path.join(settings.BASE_DIR, 'models/model_config.json')
    logger.info(f"Loading config from: {config_path}")

    if not os.path.exists(config_path):
        # Fallback hardcoded config untuk model burung jika file tidak ada
        logger.warning(f"Model config file not found at: {config_path}. Using hardcoded defaults.")
        config = {
            "sample_rate": 22050,
            "duration": 3,
            "n_mfcc": 40,
            "max_time_steps": 216,
            "num_classes": 42,
            "tflite_version": "float32"  # Default TFLite version
        }
        logger.info(f"Using fallback config: {config}")
        return config

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from file: {config}")
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        # Fallback jika file ada tapi error saat parsing
        logger.warning("Using hardcoded config due to parsing error.")
        return {
            "sample_rate": 22050,
            "duration": 3,
            "n_mfcc": 40,
            "max_time_steps": 216,
            "num_classes": 42,
            "tflite_version": "float32"
        }


def load_class_names():
    """Load class names"""
    class_names_path = os.path.join(settings.BASE_DIR, 'models/class_names.json')
    logger.info(f"Loading class names from: {class_names_path}")

    if not os.path.exists(class_names_path):
        logger.error(f"Class names file not found at: {class_names_path}")
        raise FileNotFoundError(f"Class names file not found at: {class_names_path}")

    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        logger.info(f"Loaded {len(class_names)} class names")
        return class_names
    except Exception as e:
        logger.error(f"Error loading class names: {e}")
        raise


def load_model(tflite_version='int8', model_path=None):
    """Load TFLite model dengan tflite-runtime (hanya TFLite support)"""
    if model_path is None:
        base_path = os.path.join(settings.BASE_DIR, 'models')
        model_filename = f'bird_classifier_{tflite_version}.tflite'
        model_path = os.path.join(base_path, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model file not found at: {model_path}")

    try:
        logger.info(f"Loading TFLite model from: {model_path} (version: {tflite_version})")
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info(
            f"TFLite model loaded successfully with tflite-runtime. Input: {input_details[0]['shape']}, Output: {output_details[0]['shape']}")
        return interpreter, input_details, output_details  # Tuple: (interpreter, in_details, out_details)
    except Exception as e:
        logger.error(f"Failed to load TFLite model: {e}")
        raise Exception(f"Failed to load TFLite model: {e}")


def preprocess_audio(audio_path, sr=22050, duration=3):
    """
    Preprocessing file audio (integrated dari kode preprocessing Anda)
    """
    try:
        # Load audio file dengan parameter tambahan
        audio, sr = librosa.load(audio_path, sr=sr, res_type='kaiser_fast', duration=duration)
        if len(audio) == 0:
            logger.warning(f"Empty audio file - {audio_path}")
            return None

        # Normalisasi amplitudo (-1 to 1)
        normalized_audio = librosa.util.normalize(audio)

        # Potong atau pad ke durasi yang diinginkan
        target_length = sr * duration
        if len(normalized_audio) > target_length:
            normalized_audio = normalized_audio[:target_length]
        else:
            normalized_audio = librosa.util.fix_length(normalized_audio, size=target_length)

        return normalized_audio
    except Exception as e:
        logger.error(f"Error preprocessing {audio_path}: {str(e)}")
        return None


def extract_mfcc_features(audio, config):
    """Ekstrak MFCC dari audio (sesuai dengan model TFLite yang ada)"""
    try:
        logger.info(f"Extracting MFCC with config: {config}")

        # Validasi config parameters
        required_params = ['sample_rate', 'n_mfcc', 'max_time_steps']
        for param in required_params:
            if param not in config:
                logger.error(f"Missing required config parameter: {param}")
                return None

        sample_rate = config['sample_rate']
        n_mfcc = config['n_mfcc']
        max_time_steps = config['max_time_steps']

        logger.info(f"MFCC params: sr={sample_rate}, n_mfcc={n_mfcc}, max_time_steps={max_time_steps}")
        logger.info(f"Audio shape: {audio.shape}")

        # Ekstrak MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc
        )

        logger.info(f"Raw MFCC shape: {mfcc.shape}")

        # Padding atau crop untuk max_time_steps
        if mfcc.shape[1] < max_time_steps:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_time_steps - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_time_steps]

        logger.info(f"Padded/cropped MFCC shape: {mfcc.shape}")

        # Transpose untuk format (time_steps, n_mfcc) -> (max_time_steps, n_mfcc)
        result = mfcc.T
        logger.info(f"Final MFCC shape: {result.shape}")

        return result

    except Exception as e:
        logger.error(f"Error extracting MFCC: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def normalize_features(feat, config):
    """Normalisasi MFCC menggunakan mean dan std dari normalization_params.npz"""
    try:
        norm_params_path = os.path.join(settings.BASE_DIR, 'models/normalization_params.npz')
        logger.info(f"Loading normalization params from: {norm_params_path}")

        if not os.path.exists(norm_params_path):
            logger.error(f"Normalization params file not found at: {norm_params_path}")
            raise FileNotFoundError(f"Normalization params file not found at: {norm_params_path}")

        norm_params = np.load(norm_params_path)
        mean = norm_params['mean']
        std = norm_params['std']

        logger.info(f"Loaded normalization params - mean shape: {mean.shape}, std shape: {std.shape}")
        logger.info(f"Feature shape before normalization: {feat.shape}")

        # Normalisasi menggunakan mean dan std
        feat_norm = (feat - mean) / (std + 1e-8)

        logger.info(f"Feature shape after normalization: {feat_norm.shape}")
        return feat_norm

    except Exception as e:
        logger.warning(f"Error loading normalization params: {e}. Using zero mean normalization.")
        # Fallback ke zero mean normalization
        feat_norm = (feat - np.mean(feat)) / (np.std(feat) + 1e-8)
        logger.info(f"Used fallback normalization - feature shape: {feat_norm.shape}")
        return feat_norm


def predict_with_model(interpreter, input_details, output_details, mfcc_input, tflite_version='float32'):
    """Prediksi dengan TFLite model (tflite-runtime)"""
    try:
        logger.info(f"TFLite inference (version: {tflite_version})")

        # Input sudah dalam format yang benar (normalized MFCC)
        input_data = mfcc_input.astype(input_details[0]['dtype'])

        # Handle quantization untuk model int8
        if tflite_version == 'int8' and input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = input_scale * input_data + input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Handle quantization untuk output jika model int8
        if tflite_version == 'int8' and output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            predictions = output_scale * (output_data.astype(np.float32) - output_zero_point)
        else:
            predictions = output_data.astype(np.float32)

        logger.info("TFLite prediction completed")
        return predictions.flatten()
    except Exception as e:
        logger.error(f"Error in predict_with_model: {e}")
        raise


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


def predict_single_audio(audio_file, tflite_version='float32', model=None, config=None, class_names=None):
    """
    Predict bird species from audio file using TFLite (tflite-runtime)
    """
    audio_path = None
    try:
        logger.info(
            f"Starting prediction for audio file: {getattr(audio_file, 'name', 'unknown')} (TFLite version: {tflite_version})")
        logger.info(f"BASE_DIR: {settings.BASE_DIR}")
        logger.info(f"Current working directory: {os.getcwd()}")

        # Validasi input
        if not audio_file:
            logger.error("No audio file provided")
            return {"error": "No audio file provided"}

        # Load config jika belum ada
        if config is None:
            config = load_model_config()

        # Load class names jika belum ada
        if class_names is None:
            class_names = load_class_names()

        # Load model jika belum ada
        if model is None:
            model = load_model(tflite_version)

        # Unpack model
        interpreter, input_details, output_details = model
        logger.info("TFLite model unpacked successfully")

        # Simpan file audio
        audio_path = save_audio_tempfile(audio_file)
        if not audio_path:
            return {"error": "Failed to save audio file"}

        # Preprocess audio
        audio = preprocess_audio(audio_path, config['sample_rate'], config['duration'])
        if audio is None:
            return {"error": "Failed to preprocess audio file"}

        # Extract MFCC features
        mfcc_features = extract_mfcc_features(audio, config)
        if mfcc_features is None:
            return {"error": "Failed to extract MFCC features"}

        # Normalisasi
        mfcc_norm = normalize_features(mfcc_features, config)
        mfcc_input = np.expand_dims(mfcc_norm, axis=0)  # (1, max_time_steps, n_mfcc)

        logger.info(f"Input shape for model: {mfcc_input.shape}")

        # Predict
        predictions = predict_with_model(interpreter, input_details, output_details, mfcc_input, tflite_version)

        # Get top prediction
        top_index = np.argmax(predictions)
        prediction = class_names[top_index]
        confidence = float(predictions[top_index])

        logger.info(f"TFLite prediction: {prediction}, Confidence: {confidence}")

        # Ambil bird data dari DB
        bird = Bird.objects.filter(scientific_nm=prediction).first()
        if not bird:
            logger.warning(f"No bird found for: {prediction}")
            return {
                'scientific_nm': prediction,
                'confidence': confidence,
                'bird_data': None,
                'method': f'tflite_{tflite_version}_model'
            }

        bird_data = BirdDetailSerializer(bird).data
        return {
            'scientific_nm': prediction,
            'confidence': confidence,
            'bird_data': bird_data,
            'method': f'tflite_{tflite_version}_model'
        }

    except Exception as e:
        logger.error(f"Unexpected error in predict_single_audio: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Unexpected error: {str(e)}"}

    finally:
        # Cleanup
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
                logger.info(f"Cleaned up: {audio_path}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")


def predict_from_file_path(file_path, tflite_version='float32', model=None, config=None, class_names=None, use_cpu=False):
    """
    Predict from file path using TFLite (tflite-runtime)
    """
    try:
        logger.info(f"Starting prediction for: {file_path} (TFLite version: {tflite_version})")

        if use_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        if config is None:
            config = load_model_config()
        if class_names is None:
            class_names = load_class_names()
        if model is None:
            model = load_model(tflite_version)

        # Unpack model
        interpreter, input_details, output_details = model
        logger.info("TFLite model unpacked successfully")

        # Preprocess
        audio = preprocess_audio(file_path, config['sample_rate'], config['duration'])
        if audio is None:
            return {"error": "Failed to preprocess audio"}

        # Extract MFCC features
        mfcc_features = extract_mfcc_features(audio, config)
        if mfcc_features is None:
            return {"error": "Failed to extract MFCC features"}

        # Normalisasi
        mfcc_norm = normalize_features(mfcc_features, config)
        mfcc_input = np.expand_dims(mfcc_norm, axis=0)  # (1, max_time_steps, n_mfcc)

        logger.info(f"Input shape for model: {mfcc_input.shape}")

        # Predict
        predictions = predict_with_model(interpreter, input_details, output_details, mfcc_input, tflite_version)

        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'class': class_names[idx],
                'confidence': float(predictions[idx])
            })

        logger.info(f"Top prediction: {results[0]['class']} (confidence: {results[0]['confidence']:.3f})")
        return results

    except Exception as e:
        logger.error(f"Unexpected error in predict_from_file_path: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Unexpected error: {str(e)}"}


def predict_batch_files(audio_files, tflite_version='float32', model=None, config=None, class_names=None, use_cpu=False):
    """
    Predict batch using TFLite (tflite-runtime)
    """
    results = {}

    # Load model sekali di awal
    if model is None:
        model = load_model(tflite_version)
        interpreter, input_details, output_details = model

    if config is None:
        config = load_model_config()
    if class_names is None:
        class_names = load_class_names()

    for audio_file in audio_files:
        logger.info(f"Processing: {audio_file}")
        predictions = predict_from_file_path(audio_file, tflite_version, (interpreter, input_details, output_details),
                                             config, class_names, use_cpu)

        if isinstance(predictions, list):
            results[audio_file] = predictions
            logger.info(f"  Predicted: {predictions[0]['class']} (confidence: {predictions[0]['confidence']:.3f})")
        else:
            results[audio_file] = None
            logger.error(f"  Error processing file: {predictions.get('error', 'Unknown error')}")

    return results


def main():
    """
    Main function for command line interface (TFLite only dengan tflite-runtime)
    """
    import argparse

    parser = argparse.ArgumentParser(description='Predict bird species from audio files using TFLite (tflite-runtime)')
    parser.add_argument('audio_path', type=str, help='Path to audio file or directory')
    parser.add_argument('--tflite_version', type=str, choices=['float32', 'int8'], default='int8',
                        help='TFLite version: float32 or int8 (default: int8)')
    parser.add_argument('--model', type=str, default=None,
                        help='Custom path to TFLite model (default: auto-detect based on tflite_version)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions (default: 0.5)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage (default)')

    args = parser.parse_args()

    # Force CPU if requested (sudah default)
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Running on CPU...")

    # Determine model path
    if args.model is None:
        base_path = os.path.join(settings.BASE_DIR, 'models') if 'settings' in globals() else 'models'
        model_path = os.path.join(base_path, f'bird_species_model_{args.tflite_version}.tflite')
    else:
        model_path = args.model

    # Load model
    try:
        model = load_model(args.tflite_version, model_path)
        print(f"Loaded TFLite model from {model_path} (version: {args.tflite_version})...")
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        print(traceback.format_exc())
        return

    # Load config and class names
    try:
        config = load_model_config()
        class_names = load_class_names()
        print("Config and class names loaded successfully")
    except Exception as e:
        print(f"Error loading config or class names: {e}")
        print(traceback.format_exc())
        return

    # Check if input is file or directory
    if os.path.isfile(args.audio_path):
        # Single file prediction
        print(f"\nPredicting for: {args.audio_path}")
        predictions = predict_from_file_path(args.audio_path, args.tflite_version, model, config, class_names, args.cpu)

        if isinstance(predictions, list):
            print("\nPrediction Results (Top 3):")
            print("-" * 50)
            for i, pred in enumerate(predictions):
                print(f"{i + 1}. {pred['class']}: {pred['confidence']:.3f}")
                if pred['confidence'] < args.threshold:
                    print(f"   (Below threshold {args.threshold})")
        else:
            print(f"Error processing audio file: {predictions.get('error', 'Unknown error')}")

    elif os.path.isdir(args.audio_path):
        # Batch prediction for directory
        audio_files = []
        for file in os.listdir(args.audio_path):
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(args.audio_path, file))

        if not audio_files:
            print(f"No audio files found in {args.audio_path}")
            return

        print(f"\nFound {len(audio_files)} audio files")
        results = predict_batch_files(audio_files, args.tflite_version, model, config, class_names, args.cpu)

        # Save results to JSON
        output_file = 'prediction_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_file}")
        print("Sample results:")
        for filename, preds in list(results.items())[:3]:  # Tampilkan 3 pertama
            if preds:
                print(f"  {os.path.basename(filename)}: {preds[0]['class']} ({preds[0]['confidence']:.3f})")

    else:
        print(f"Error: '{args.audio_path}' is not a valid file or directory")


if __name__ == "__main__":
    main()
