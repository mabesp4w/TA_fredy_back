import joblib
import pandas as pd
from api.utils.preprocessing_audio import preprocess_audio, extract_features

def load_model(model_path='api/utils/random_forest_model.joblib'):
    return joblib.load(model_path)

def predict_single_audio(audio_path, model=None):
    if model is None:
        model = load_model()
    
    # Preprocess audio
    processed_audio = preprocess_audio(audio_path)
    if processed_audio is None:
        return None
    
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
    
    return {
        'predicted_species': prediction,
        'confidence': max(probabilities)
    }