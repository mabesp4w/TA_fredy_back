import joblib
import pandas as pd
import tempfile
import os
import logging
from django.conf import settings
from api.utils.preprocessing_audio import preprocess_audio, extract_features
from crud.serializers import BirdDetailSerializer
from crud.models import Bird

logger = logging.getLogger(__name__)


def load_model(model_path=None):
    """Load the trained model with proper path handling"""
    if model_path is None:
        # Gunakan absolute path berdasarkan BASE_DIR Django
        model_path = os.path.join(settings.BASE_DIR, 'api/utils/random_forest_model.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    return joblib.load(model_path)


def save_audio_tempfile(audio_file):
    """
    Menyimpan file audio yang diupload ke dalam file sementara di disk
    """
    try:
        # Pastikan kita mendapatkan file extension yang benar
        original_name = getattr(audio_file, 'name', 'audio.wav')
        file_extension = os.path.splitext(original_name)[1] or '.wav'

        # Gunakan directory temp yang aman
        temp_dir = getattr(settings, 'TEMP_DIR', tempfile.gettempdir())

        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_extension,
                dir=temp_dir,
                prefix='audio_'
        ) as temp_file:
            # Reset file pointer jika perlu
            if hasattr(audio_file, 'seek'):
                audio_file.seek(0)

            # Write file chunks
            for chunk in audio_file.chunks():
                temp_file.write(chunk)

            temp_file.flush()  # Ensure data is written to disk
            os.fsync(temp_file.fileno())  # Force write to disk

            temp_path = temp_file.name

        # Verify file was created and has content
        if not os.path.exists(temp_path):
            logger.error(f"Temporary file was not created: {temp_path}")
            return None

        file_size = os.path.getsize(temp_path)
        if file_size == 0:
            logger.error(f"Temporary file is empty: {temp_path}")
            os.unlink(temp_path)
            return None

        logger.info(f"Audio file saved to: {temp_path} (size: {file_size} bytes)")
        return temp_path

    except Exception as e:
        logger.error(f"Error creating temp file: {e}")
        return None


def predict_single_audio(audio_file, model=None):
    """
    Predict bird species from audio file
    """
    audio_path = None
    try:
        logger.info(f"Starting prediction for audio file: {getattr(audio_file, 'name', 'unknown')}")

        # Validasi input
        if not audio_file:
            logger.error("No audio file provided")
            return {"error": "No audio file provided"}

        # Simpan file audio yang diupload ke file sementara
        audio_path = save_audio_tempfile(audio_file)
        if not audio_path:
            logger.error("Failed to save audio file to temporary location")
            return {"error": "Failed to save audio file"}

        logger.info(f"Audio saved to temporary path: {audio_path}")

        # Load model jika belum ada
        if model is None:
            try:
                model = load_model()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return {"error": f"Failed to load model: {str(e)}"}

        # Preprocess audio - PASTIKAN audio_path adalah string
        logger.info(f"Preprocessing audio from path: {audio_path}")
        logger.info(f"Path type: {type(audio_path)}")

        try:
            processed_audio = preprocess_audio(audio_path)
            if processed_audio is None:
                logger.error("Audio preprocessing returned None")
                return {"error": "Failed to preprocess audio"}
            logger.info("Audio preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error during audio preprocessing: {e}")
            return {"error": f"Failed to preprocess audio: {str(e)}"}

        # Extract features
        try:
            features = extract_features(processed_audio)
            if features is None:
                logger.error("Feature extraction returned None")
                return {"error": "Failed to extract features"}
            logger.info("Feature extraction completed successfully")
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            return {"error": f"Failed to extract features: {str(e)}"}

        # Convert to DataFrame dengan struktur yang sama
        try:
            features_df = pd.DataFrame([features])

            # Pastikan urutan kolom sama dengan training
            feature_cols = [col for col in model.feature_names_in_]
            features_df = features_df[feature_cols]

            logger.info(f"Features DataFrame shape: {features_df.shape}")
        except Exception as e:
            logger.error(f"Error creating features DataFrame: {e}")
            return {"error": f"Failed to prepare features: {str(e)}"}

        # Predict
        try:
            prediction = model.predict(features_df)[0]
            probabilities = model.predict_proba(features_df)[0]
            confidence = max(probabilities)

            logger.info(f"Prediction: {prediction}, Confidence: {confidence}")
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
                    'bird_data': None
                }

            # Serialize the bird data using the BirdDetailSerializer
            bird_data = BirdDetailSerializer(bird).data

            logger.info(f"Successfully completed prediction for {prediction}")
            return {
                'scientific_nm': prediction,
                'confidence': confidence,
                'bird_data': bird_data
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