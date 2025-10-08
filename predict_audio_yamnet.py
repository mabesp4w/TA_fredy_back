#!/usr/bin/env python
"""
Script untuk prediksi file audio menggunakan YAMNet model yang sudah di-train
Mendukung berbagai format audio dan memberikan prediksi dengan confidence score
Versi CPU-only untuk mengatasi masalah CUDA
"""
import os
import sys
import numpy as np
import librosa
import keras
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

# Force CPU usage untuk mengatasi masalah CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class YAMNetAudioPredictor:
    """Kelas untuk prediksi audio menggunakan YAMNet model yang sudah di-train"""
    
    def __init__(self, model_path='models/bird_sound_classifier.h5', config_path='models/model_config.json', class_names_path='models/class_names.json'):
        """
        Inisialisasi predictor
        
        Args:
            model_path (str): Path ke model yang sudah di-save
            config_path (str): Path ke config model
            class_names_path (str): Path ke class names
        """
        self.model_path = model_path
        self.model = None
        self.class_names = None
        self.config = None
        
        # Load model dan metadata
        self.load_model()
        self.load_config(config_path)
        self.load_class_names(class_names_path)
        
    def load_model(self):
        """Load model yang sudah di-train"""
        try:
            self.model = keras.models.load_model(self.model_path, compile=False)
            print(f"Model berhasil di-load dari: {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def load_config(self, config_path):
        """Load model configuration"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Config loaded: {self.config['n_classes']} classes")
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = None
    
    def load_class_names(self, class_names_path):
        """Load class names"""
        try:
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Class names loaded: {len(self.class_names)} classes")
            print(f"Classes: {self.class_names}")
        except Exception as e:
            print(f"Error loading class names: {e}")
            self.class_names = None
    
    def preprocess_audio(self, audio_path):
        """
        Preprocess audio file untuk prediksi
        
        Args:
            audio_path (str): Path ke file audio
            
        Returns:
            np.array: Preprocessed audio features
        """
        try:
            # Load audio file
            audio, sr = librosa.load(
                audio_path, 
                sr=self.config['sample_rate'], 
                duration=self.config['duration']
            )
            
            # Normalisasi audio
            audio = librosa.util.normalize(audio)
            
            # Pad atau trim audio ke durasi yang diinginkan
            target_length = int(self.config['sample_rate'] * self.config['duration'])
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.config['sample_rate'],
                n_mels=self.config['n_mels'],
                n_fft=self.config['n_fft'],
                hop_length=self.config['hop_length']
            )
            
            # Konversi ke dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Error preprocessing audio {audio_path}: {e}")
            return None
    
    def predict_single(self, audio_path, top_k=3):
        """
        Prediksi untuk single audio file
        
        Args:
            audio_path (str): Path ke file audio
            top_k (int): Jumlah prediksi teratas yang ditampilkan
            
        Returns:
            dict: Hasil prediksi
        """
        if self.model is None:
            print("Model belum di-load!")
            return None
        
        # Preprocess audio
        features = self.preprocess_audio(audio_path)
        if features is None:
            return None
        
        # Reshape untuk prediction
        features_reshaped = np.expand_dims(features, axis=0)
        features_reshaped = features_reshaped[..., np.newaxis]  # Add channel dimension
        
        # Prediksi
        try:
            prediction = self.model.predict(features_reshaped, verbose=0)
        except Exception as e:
            print(f"Error saat prediksi: {e}")
            return None
        
        # Ambil top-k predictions
        top_k_indices = np.argsort(prediction[0])[-top_k:][::-1]
        
        results = {
            'file_path': audio_path,
            'predictions': []
        }
        
        for i, idx in enumerate(top_k_indices):
            class_name = self.class_names[idx] if self.class_names else f'Class {idx}'
            confidence = prediction[0][idx]
            
            results['predictions'].append({
                'rank': i + 1,
                'class': class_name,
                'confidence': confidence
            })
        
        return results
    
    def predict_batch(self, audio_paths, top_k=3):
        """
        Prediksi untuk multiple audio files
        
        Args:
            audio_paths (list): List path ke file audio
            top_k (int): Jumlah prediksi teratas yang ditampilkan
            
        Returns:
            list: List hasil prediksi
        """
        results = []
        
        print(f"Memproses {len(audio_paths)} file audio...")
        
        for audio_path in audio_paths:
            result = self.predict_single(audio_path, top_k)
            if result:
                results.append(result)
        
        return results
    
    def predict_from_folder(self, folder_path, top_k=3, file_extensions=('.wav', '.mp3', '.flac', '.m4a')):
        """
        Prediksi untuk semua file audio dalam folder
        
        Args:
            folder_path (str): Path ke folder
            top_k (int): Jumlah prediksi teratas yang ditampilkan
            file_extensions (tuple): Ekstensi file yang didukung
            
        Returns:
            list: List hasil prediksi
        """
        # Cari semua file audio
        audio_files = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(file_extensions):
                audio_files.append(os.path.join(folder_path, file))
        
        if not audio_files:
            print(f"Tidak ada file audio ditemukan di {folder_path}")
            return []
        
        print(f"Ditemukan {len(audio_files)} file audio")
        return self.predict_batch(audio_files, top_k)
    
    def print_prediction_results(self, results):
        """Print hasil prediksi dengan format yang rapi"""
        if isinstance(results, dict):
            results = [results]
        
        for result in results:
            print(f"\n{'='*60}")
            print(f"File: {os.path.basename(result['file_path'])}")
            print(f"{'='*60}")
            
            for pred in result['predictions']:
                print(f"{pred['rank']}. {pred['class']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
    
    def save_predictions_to_csv(self, results, output_path='predictions.csv'):
        """
        Simpan hasil prediksi ke CSV
        
        Args:
            results (list): Hasil prediksi
            output_path (str): Path untuk menyimpan CSV
        """
        import pandas as pd
        
        # Flatten results untuk CSV
        csv_data = []
        for result in results:
            file_name = os.path.basename(result['file_path'])
            for pred in result['predictions']:
                csv_data.append({
                    'file_name': file_name,
                    'file_path': result['file_path'],
                    'rank': pred['rank'],
                    'predicted_class': pred['class'],
                    'confidence': pred['confidence'],
                    'confidence_percentage': pred['confidence'] * 100
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        print(f"Hasil prediksi disimpan ke: {output_path}")

def main():
    """Fungsi utama untuk prediksi"""
    parser = argparse.ArgumentParser(description='Prediksi suara burung menggunakan YAMNet model')
    parser.add_argument('--model', type=str, default='models/bird_sound_classifier.h5', 
                       help='Path ke model yang sudah di-train')
    parser.add_argument('--audio', type=str, help='Path ke file audio tunggal')
    parser.add_argument('--folder', type=str, help='Path ke folder berisi file audio')
    parser.add_argument('--top_k', type=int, default=3, help='Jumlah prediksi teratas')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Path untuk menyimpan hasil')
    
    args = parser.parse_args()
    
    # Inisialisasi predictor
    predictor = YAMNetAudioPredictor(args.model)
    
    if predictor.model is None:
        print("Gagal memuat model!")
        return
    
    results = []
    
    if args.audio:
        # Prediksi single file
        print(f"Memproses file: {args.audio}")
        result = predictor.predict_single(args.audio, args.top_k)
        if result:
            results.append(result)
    
    elif args.folder:
        # Prediksi dari folder
        print(f"Memproses folder: {args.folder}")
        results = predictor.predict_from_folder(args.folder, args.top_k)
    
    else:
        # Prediksi dari file test_audio.wav default
        test_audio = 'test_audio.wav'
        if os.path.exists(test_audio):
            print(f"Memproses file default: {test_audio}")
            result = predictor.predict_single(test_audio, args.top_k)
            if result:
                results.append(result)
        else:
            print("Tidak ada file audio yang ditentukan!")
            return
    
    if results:
        # Print hasil
        predictor.print_prediction_results(results)
        
        # Simpan ke CSV
        predictor.save_predictions_to_csv(results, args.output)
        
        print(f"\nPrediksi selesai! {len(results)} file diproses.")
    else:
        print("Tidak ada hasil prediksi!")

if __name__ == "__main__":
    main()
