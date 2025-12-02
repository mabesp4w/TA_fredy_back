# Bird Sound Prediction Usage Guide

## Overview
Project Django ini sekarang dapat melakukan prediksi suara burung menggunakan model TensorFlow/Keras yang sama seperti project MLBurung. Model menggunakan mel-spectrogram sebagai feature extraction dan dapat mengenali 10 jenis burung.

## Model Information
- **Model File**: `api/utils/model/bird_sound_classifier.h5`
- **Config File**: `api/utils/model/model_config.json`
- **Class Names**: `api/utils/model/class_names.json`
- **Supported Classes**: 10 jenis burung (Aegithina tiphia, Aplonis panayensis, Cacatua galerita, Eclectus roratus, Geoffroyus geoffroyi, Geopelia striata, Orthotomus sutorius, Pycnonotus aurigaster, Rhyticeros plicatus, Tanysiptera galatea)

## Usage Methods

### 1. Django API Integration
File `api/utils/predict.py` telah diupdate untuk menggunakan model TensorFlow. Fungsi utama:
- `predict_single_audio(audio_file, model=None, config=None, class_names=None)`: Untuk prediksi dari file upload Django
- `predict_from_file_path(file_path, model=None, config=None, class_names=None, use_cpu=False)`: Untuk prediksi dari file path
- `predict_batch_files(audio_files, model=None, config=None, class_names=None, use_cpu=False)`: Untuk prediksi batch

### 2. Command Line Interface
Dapat menggunakan command line seperti project MLBurung:

```bash
# Prediksi single file
python api/utils/predict.py test_audio.wav --cpu

# Prediksi dengan threshold custom
python api/utils/predict.py test_audio.wav --cpu --threshold 0.7

# Prediksi batch dari directory
python api/utils/predict.py /path/to/audio/directory --cpu
```

### 3. Standalone Testing
Untuk testing tanpa Django context, gunakan script standalone:

```bash
python test_predict_standalone.py test_audio.wav --cpu
```

## Key Features
- **CPU/GPU Support**: Dapat memaksa penggunaan CPU dengan flag `--cpu`
- **Batch Processing**: Mendukung prediksi multiple files sekaligus
- **Top 3 Predictions**: Menampilkan 3 prediksi terbaik dengan confidence score
- **Threshold Filtering**: Filter prediksi berdasarkan confidence threshold
- **JSON Output**: Hasil batch dapat disimpan ke file JSON

## Example Output
```
Prediction Results:
--------------------------------------------------
1. Eclectus roratus: 0.699
2. Cacatua galerita: 0.140
3. Geoffroyus geoffroyi: 0.061
```

## Dependencies
- TensorFlow 2.19.0
- librosa 0.10.1
- numpy 1.26.4
- Django 4.2.7 (untuk integrasi API)

## Notes
- Model menggunakan mel-spectrogram dengan konfigurasi: sample_rate=22050, duration=5s, n_mels=128
- File audio akan di-pad atau di-crop ke durasi 5 detik
- Model dioptimalkan untuk file audio dengan format WAV/MP3
- Confidence score menunjukkan tingkat keyakinan prediksi (0.0 - 1.0)
