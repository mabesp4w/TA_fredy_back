import os

import joblib
import pandas as pd
from django.conf import settings

from api.utils.preprocessing_audio import preprocess_audio, extract_features
from crud.serializers import BirdDetailSerializer
from crud.models import Bird
import tempfile
import logging

logger = logging.getLogger(__name__)


def load_model(model_path=None):
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
        # Gunakan directory yang lebih spesifik
        temp_dir = getattr(settings, 'TEMP_DIR', '/tmp')

        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                dir=temp_dir
        ) as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_file.flush()  # Ensure data is written
            return temp_file.name
    except Exception as e:
        print(f"Error creating temp file: {e}")
        return None

def predict_single_audio(audio_file, model=None):
    audio_path = None
    try:
        logger.info(f"Processing audio file: {audio_file.name}")

        # Simpan file audio yang diupload ke file sementara
        audio_path = save_audio_tempfile(audio_file)
        if not audio_path:
            logger.error("Failed to save audio file")
            return {"error": "Failed to save audio file"}

        logger.info(f"Audio saved to: {audio_path}")

        if model is None:
            model = load_model()
            logger.info("Model loaded successfully")

        # Preprocess audio
        processed_audio = preprocess_audio(audio_path)
        if processed_audio is None:
            logger.error("Failed to preprocess audio")
            return {"error": "Failed to preprocess audio"}

        logger.info("Audio preprocessing completed")

        # ... rest of your code ...

    except Exception as e:
        logger.error(f"Error in predict_single_audio: {str(e)}")
        return {"error": str(e)}
    finally:
        # Cleanup temporary file
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
