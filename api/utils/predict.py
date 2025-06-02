import joblib
import pandas as pd
from api.utils.preprocessing_audio import preprocess_audio, extract_features
from crud.serializers import BirdDetailSerializer
from crud.models import Bird
import tempfile


def load_model(model_path='api/utils/random_forest_model.joblib'):
    return joblib.load(model_path)

def save_audio_tempfile(audio_file):
    """
    Menyimpan file audio yang diupload ke dalam file sementara di disk
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        for chunk in audio_file.chunks():
            temp_file.write(chunk)
        return temp_file.name  # Mengembalikan path sementara file audio

def predict_single_audio(audio_file, model=None):
    try:
        # Simpan file audio yang diupload ke file sementara
        audio_path = save_audio_tempfile(audio_file)
        if not audio_path:
            return {"error": "Failed to save audio file"}

        if model is None:
            model = load_model()

        # Preprocess audio
        processed_audio = preprocess_audio(audio_path)
        if processed_audio is None:
            return {"error": "Failed to preprocess audio"}

        # Extract features
        features = extract_features(processed_audio)

        # Convert to DataFrame dengan struktur yang sama
        features_df = pd.DataFrame([features])

        # Pastikan urutan kolom sama dengan training
        feature_cols = [col for col in model.feature_names_in_]
        features_df = features_df[feature_cols]

        # Predict
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]

        # Ambil bird berdasarkan scientific_nm yang diprediksi
        bird = Bird.objects.filter(scientific_nm=prediction).first()
        if not bird:
            return {
                'scientific_nm': prediction,
                'confidence': max(probabilities),
                'bird_data': None
            }

        # Serialize the bird data using the BirdDetailSerializer
        bird_data = BirdDetailSerializer(bird).data

        return {
            'scientific_nm': prediction,
            'confidence': max(probabilities),
            'bird_data': bird_data  # Include bird data in the response
        }

    except Exception as e:
        return {"error": str(e)}
