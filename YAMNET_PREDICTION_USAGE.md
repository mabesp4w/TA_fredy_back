# YAMNet Model Prediction Usage

## Overview
Script `predict_audio_yamnet.py` adalah implementasi prediksi suara burung menggunakan model YAMNet yang sudah di-train dengan data augmentation. Script ini menggantikan `predict_audio_cpu.py` dari project YAMNet_fine-tuning dan terintegrasi dengan Django project.

## Features
- ✅ **CPU-only mode** - Menghindari masalah CUDA/GPU compatibility
- ✅ **Multiple audio formats** - Mendukung .wav, .mp3, .flac, .m4a
- ✅ **Batch processing** - Bisa memproses folder berisi multiple files
- ✅ **Top-K predictions** - Menampilkan beberapa prediksi teratas
- ✅ **CSV export** - Menyimpan hasil ke file CSV
- ✅ **Django integration** - Terintegrasi dengan `api/utils/predict.py`

## Model Information
- **Model**: YAMNet fine-tuned dengan data augmentation
- **Classes**: 10 species burung
- **Input**: Mel-spectrogram (128x130)
- **Output**: 10-class classification
- **Framework**: Keras 3.x + TensorFlow 2.19

### Supported Bird Species
1. Aegithina tiphia
2. Aplonis panayensis  
3. Cacatua galerita
4. Eclectus roratus
5. Geoffroyus geoffroyi
6. Geopelia striata
7. Orthotomus sutorius
8. Pycnonotus aurigaster
9. Rhyticeros plicatus
10. Tanysiptera galatea

## Usage

### 1. Basic Usage (Single File)
```bash
python3 predict_audio_yamnet.py --audio test_audio.wav
```

### 2. Multiple Predictions (Top-K)
```bash
python3 predict_audio_yamnet.py --audio test_audio.wav --top_k 5
```

### 3. Batch Processing (Folder)
```bash
python3 predict_audio_yamnet.py --folder /path/to/audio/folder --top_k 3
```

### 4. Custom Output
```bash
python3 predict_audio_yamnet.py --audio test_audio.wav --output my_predictions.csv
```

### 5. Default Behavior
```bash
python3 predict_audio_yamnet.py
# Will process test_audio.wav if it exists
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `models/bird_sound_classifier.h5` | Path to model file |
| `--audio` | str | None | Path to single audio file |
| `--folder` | str | None | Path to folder containing audio files |
| `--top_k` | int | 3 | Number of top predictions to show |
| `--output` | str | `predictions.csv` | Output CSV file path |

## Output Format

### Console Output
```
============================================================
File: test_audio.wav
============================================================
1. Aegithina tiphia: 0.9915 (99.15%)
2. Tanysiptera galatea: 0.0039 (0.39%)
3. Rhyticeros plicatus: 0.0023 (0.23%)
```

### CSV Output
| file_name | file_path | rank | predicted_class | confidence | confidence_percentage |
|-----------|-----------|------|-----------------|------------|---------------------|
| test_audio.wav | /path/to/test_audio.wav | 1 | Aegithina tiphia | 0.9915 | 99.15 |
| test_audio.wav | /path/to/test_audio.wav | 2 | Tanysiptera galatea | 0.0039 | 0.39 |

## Django Integration

Script ini juga terintegrasi dengan Django project melalui `api/utils/predict.py`:

```python
from api.utils.predict import predict_single_audio, predict_from_file_path

# For Django views/API
result = predict_single_audio(audio_file)

# For file path prediction
result = predict_from_file_path("test_audio.wav", use_cpu=True)
```

## Requirements

### Dependencies
- Python 3.8+
- TensorFlow 2.19.0
- Keras 3.11.3
- librosa 0.10.1
- numpy 1.26.4
- pandas 2.1.4

### Model Files Required
```
models/
├── bird_sound_classifier.h5      # Main model file
├── model_config.json            # Model configuration
├── class_names.json             # Class names
├── metadata.pkl                 # Training metadata
└── label_encoder.pkl            # Label encoder
```

## Performance Notes

- **CPU Mode**: Script secara default menggunakan CPU untuk menghindari masalah CUDA
- **Memory Usage**: Model membutuhkan ~500MB RAM
- **Processing Time**: ~2-3 detik per file audio (3 detik duration)
- **Accuracy**: Model menunjukkan akurasi tinggi (99%+ confidence untuk prediksi yang benar)

## Troubleshooting

### Common Issues

1. **CUDA Errors**: Script sudah dikonfigurasi untuk CPU-only mode
2. **Model Loading**: Pastikan semua file model ada di folder `models/`
3. **Audio Format**: Pastikan file audio dalam format yang didukung
4. **Memory**: Untuk batch processing, pastikan RAM cukup

### Error Messages

- `Model belum di-load!`: Model file tidak ditemukan atau corrupt
- `Error preprocessing audio`: File audio tidak bisa dibaca atau format tidak didukung
- `No audio files found`: Folder tidak berisi file audio yang didukung

## Examples

### Example 1: Single File Prediction
```bash
$ python3 predict_audio_yamnet.py --audio bird_sound.wav --top_k 3
Model berhasil di-load dari: models/bird_sound_classifier.h5
Config loaded: 10 classes
Class names loaded: 10 classes
Memproses file: bird_sound.wav

============================================================
File: bird_sound.wav
============================================================
1. Aegithina tiphia: 0.9915 (99.15%)
2. Tanysiptera galatea: 0.0039 (0.39%)
3. Rhyticeros plicatus: 0.0023 (0.23%)
```

### Example 2: Batch Processing
```bash
$ python3 predict_audio_yamnet.py --folder audio_samples/ --output batch_results.csv
Model berhasil di-load dari: models/bird_sound_classifier.h5
Config loaded: 10 classes
Class names loaded: 10 classes
Ditemukan 5 file audio
Memproses 5 file audio...

============================================================
File: sample1.wav
============================================================
1. Cacatua galerita: 0.8756 (87.56%)
2. Aegithina tiphia: 0.0892 (8.92%)
3. Pycnonotus aurigaster: 0.0234 (2.34%)

[... more results ...]

Hasil prediksi disimpan ke: batch_results.csv
Prediksi selesai! 5 file diproses.
```

## Integration with Django

Untuk menggunakan dalam Django views:

```python
# views.py
from api.utils.predict import predict_single_audio

def predict_bird(request):
    if request.method == 'POST':
        audio_file = request.FILES['audio']
        result = predict_single_audio(audio_file)
        
        if 'error' not in result:
            return JsonResponse({
                'success': True,
                'prediction': result['scientific_nm'],
                'confidence': result['confidence'],
                'bird_data': result['bird_data']
            })
        else:
            return JsonResponse({'success': False, 'error': result['error']})
```

## Model Performance

- **Training**: YAMNet fine-tuned dengan data augmentation
- **Validation Accuracy**: ~95%+ pada test set
- **Inference Speed**: ~2-3 detik per file (CPU)
- **Model Size**: ~7.6 MB
- **Input Requirements**: 3 detik audio, 22050 Hz sample rate
