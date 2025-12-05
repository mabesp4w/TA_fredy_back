# Troubleshooting: Email Tidak Terkirim via Endpoint

## Masalah
- Email berhasil terkirim ketika menggunakan `send_mail()` di Django shell
- Email **TIDAK** terkirim ketika menggunakan endpoint `/auth/forgot-password/`
- Response sukses (`200 OK`) tapi email tidak sampai

## Penyebab
Container Docker belum membaca file `.env` yang baru atau belum di-restart setelah mengubah `.env`.

## Solusi

### Langkah 1: Verifikasi File `.env`

Pastikan file `.env` ada di folder `back/` dan berisi:

```env
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=smartml1990@gmail.com
EMAIL_HOST_PASSWORD=your-app-password-here
DEFAULT_FROM_EMAIL=smartml1990@gmail.com
```

**PENTING:**
- Ganti `your-app-password-here` dengan App Password Gmail Anda (16 karakter)
- Tidak ada spasi di awal/akhir setiap baris
- Tidak ada tanda kutip (`"` atau `'`) di sekitar nilai

### Langkah 2: Restart Container Docker

**WAJIB** restart container setelah mengubah `.env`:

```bash
cd back/
docker compose restart back-web-1
```

Atau jika container tidak berjalan:
```bash
docker compose up -d
```

### Langkah 3: Verifikasi Konfigurasi di Container

Masuk ke container dan cek konfigurasi:

```bash
docker exec -it back-web-1 bash
python manage.py shell
```

Lalu jalankan:
```python
from django.conf import settings
print("EMAIL_BACKEND:", settings.EMAIL_BACKEND)
print("EMAIL_HOST_USER:", settings.EMAIL_HOST_USER)
print("EMAIL_HOST_PASSWORD:", "SET" if settings.EMAIL_HOST_PASSWORD else "NOT SET")
```

**Pastikan:**
- `EMAIL_BACKEND` = `django.core.mail.backends.smtp.EmailBackend` (bukan console)
- `EMAIL_HOST_PASSWORD` = `SET` (bukan `NOT SET`)

### Langkah 4: Test Endpoint dan Cek Log

1. **Test endpoint** via Postman atau browser:
   ```
   POST http://localhost:8103/auth/forgot-password/
   Content-Type: application/json
   
   {
     "email": "pokevhee@gmail.com"
   }
   ```

2. **Cek log container** untuk melihat output debugging:
   ```bash
   docker logs back-web-1 --tail 100
   ```

   Anda akan melihat output seperti ini jika konfigurasi benar:
   ```
   ============================================================
   FORGOT PASSWORD - EMAIL CONFIGURATION
   ============================================================
   EMAIL_BACKEND: django.core.mail.backends.smtp.EmailBackend
   EMAIL_HOST: smtp.gmail.com
   EMAIL_HOST_USER: smartml1990@gmail.com
   EMAIL_HOST_PASSWORD: SET
   ...
   Email send result: 1
   ✅ Email successfully sent to pokevhee@gmail.com
   ```

   Jika masih menggunakan console backend, Anda akan melihat:
   ```
   EMAIL_BACKEND: django.core.mail.backends.console.EmailBackend
   EMAIL_HOST_PASSWORD: NOT SET (MASALAH!)
   ERROR: EMAIL_BACKEND masih menggunakan console backend!
   ```

### Langkah 5: Jika Masih Menggunakan Console Backend

Jika setelah restart container masih menggunakan console backend:

1. **Cek apakah file `.env` ada di dalam container**:
   ```bash
   docker exec -it back-web-1 ls -la /app/.env
   ```

2. **Jika file tidak ada**, copy file `.env` ke container:
   ```bash
   docker cp back/.env back-web-1:/app/.env
   docker compose restart back-web-1
   ```

3. **Atau mount file `.env` di `docker-compose.yml`**:
   ```yaml
   services:
     back-web:
       # ... other config ...
       volumes:
         - ./.env:/app/.env
   ```
   Lalu restart:
   ```bash
   docker compose up -d
   ```

## Checklist

- [ ] File `.env` ada di folder `back/`
- [ ] File `.env` berisi semua konfigurasi email dengan benar
- [ ] Container Docker sudah di-restart setelah mengubah `.env`
- [ ] Verifikasi di shell menunjukkan `EMAIL_BACKEND` = SMTP (bukan console)
- [ ] Verifikasi di shell menunjukkan `EMAIL_HOST_PASSWORD` = SET
- [ ] Test endpoint dan cek log menunjukkan konfigurasi SMTP
- [ ] Email berhasil terkirim dan sampai ke inbox

## Catatan

- **Perbedaan Shell vs Endpoint**: Ketika Anda menjalankan `send_mail()` di shell, Django membaca `.env` saat shell dimulai. Ketika endpoint dipanggil, Django sudah berjalan dan mungkin belum membaca `.env` yang baru jika container belum di-restart.
- **Restart Wajib**: Setiap kali mengubah file `.env`, container **WAJIB** di-restart agar perubahan terbaca.

