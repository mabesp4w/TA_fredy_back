# Petunjuk Menjalankan Project di Windows dengan Docker Desktop

## Prerequisites

1. **Install Docker Desktop**
   - Download dari: https://www.docker.com/products/docker-desktop/
   - Install dan jalankan Docker Desktop
   - Pastikan status "Running"

2. **Struktur Project**
   Pastikan struktur direktori seperti ini:
   ```
   parent-directory/
   ├── DjangoProject/
   │   ├── api/
   │   ├── authentication/
   │   ├── crud/
   │   ├── fredy/
   │   ├── docker-compose.yml
   │   ├── Dockerfile
   │   └── requirements.txt
   └── MLBurung/
       ├── model_20251007_063007/
       │   ├── bird_sound_classifier.h5
       │   ├── class_names.json
       │   └── model_config.json
       └── predict.py
   ```

## Langkah-langkah Setup

### 1. Buka Command Prompt atau PowerShell
- Tekan `Win + R`, ketik `cmd` atau `powershell`
- Navigate ke direktori project:
  ```cmd
  cd C:\path\to\DjangoProject
  ```

### 2. Jalankan Docker Compose
```cmd
# Build dan start containers
docker compose up -d

# Atau jika ada masalah, rebuild dari awal:
docker compose down
docker compose build --no-cache
docker compose up -d
```

### 3. Tunggu Container Ready
```cmd
# Check status containers
docker compose ps

# Check logs jika ada masalah
docker compose logs web
```

## Testing API

### 1. Test dengan curl (jika tersedia)
```cmd
curl -X POST http://localhost:8103/api/prediction/ -F "audio_file=@test_audio.wav"
```

### 2. Test dengan PowerShell
```powershell
$filePath = "C:\path\to\test_audio.wav"
$uri = "http://localhost:8103/api/prediction/"

$form = @{
    audio_file = Get-Item $filePath
}

Invoke-RestMethod -Uri $uri -Method Post -Form $form
```

### 3. Test dengan Postman/Insomnia
- **URL**: http://localhost:8103/api/prediction/
- **Method**: POST
- **Body**: form-data
- **Key**: audio_file, Type: File, Value: pilih file audio

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| **Django API** | http://localhost:8103 | Main API |
| **Prediction** | http://localhost:8103/api/prediction/ | Bird sound prediction |
| **phpMyAdmin** | http://localhost:8080 | Database management |
| **Admin Panel** | http://localhost:8103/admin/ | Django admin |

## Troubleshooting

### Port Already in Use
```cmd
# Check port yang digunakan
netstat -ano | findstr :8103
netstat -ano | findstr :3309
netstat -ano | findstr :8080

# Kill process jika perlu
taskkill /PID <PID_NUMBER> /F
```

### Docker Container Issues
```cmd
# Stop semua containers
docker compose down

# Remove semua containers dan images
docker compose down --rmi all

# Rebuild dari awal
docker compose build --no-cache
docker compose up -d
```

### Memory Issues
- Buka Docker Desktop Settings
- Go to Resources > Advanced
- Increase Memory limit (minimal 4GB)
- Restart Docker Desktop

### MLBurung Path Issues
Pastikan path MLBurung benar di `docker-compose.yml`:
```yaml
volumes:
  - ../MLBurung:/app/../MLBurung
```

## Expected Response

Jika berhasil, Anda akan mendapat response seperti ini:
```json
{
  "scientific_nm": "Eclectus roratus",
  "confidence": 0.699,
  "all_predictions": [
    {"class": "Eclectus roratus", "confidence": 0.699},
    {"class": "Cacatua galerita", "confidence": 0.14},
    {"class": "Geoffroyus geoffroyi", "confidence": 0.061}
  ],
  "method": "mlburung_original",
  "bird_data": {
    "bird_nm": "Nuri Bayan",
    "scientific_nm": "Eclectus roratus",
    "description": "Nuri besar dengan dimorfisme seksual yang ekstrem...",
    "habitat": "Hutan hujan dan daerah terbuka di Indonesia timur..."
  }
}
```

## Tips untuk Windows

1. **Gunakan WSL2**: Docker Desktop di Windows lebih baik dengan WSL2 backend
2. **File Path**: Gunakan forward slash `/` dalam docker-compose.yml
3. **Antivirus**: Whitelist Docker Desktop dan project folder di antivirus
4. **Firewall**: Allow Docker Desktop melalui Windows Firewall
5. **Run as Administrator**: Jika ada masalah permission, jalankan Command Prompt sebagai Administrator

## Database Access

- **Host**: localhost:3309
- **Username**: django_user
- **Password**: fredypwd
- **Database**: fredy
- **phpMyAdmin**: http://localhost:8080

## Support

Jika mengalami masalah:
1. Pastikan Docker Desktop running
2. Check file paths dan permissions
3. Ensure MLBurung project accessible
4. Check Windows Firewall settings
5. Verify port availability
