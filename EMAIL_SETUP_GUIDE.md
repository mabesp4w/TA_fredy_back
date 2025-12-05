<!-- @format -->

# Panduan Setup Email Gmail untuk Django

Panduan lengkap untuk mengatur Gmail agar bisa mengirim email dari aplikasi Django.

## 📋 Prasyarat

1. Akun Gmail aktif
2. Akses ke file `.env` di folder `back/`
3. Akses ke Google Account Settings

---

## 🔐 Langkah 1: Aktifkan 2-Step Verification

1. Buka [Google Account Security](https://myaccount.google.com/security)
2. Login dengan akun Gmail Anda
3. Cari bagian **"2-Step Verification"**
4. Klik **"Get Started"** atau **"Turn On"**
5. Ikuti langkah-langkah untuk mengaktifkan 2-Step Verification
   - Pilih metode verifikasi (SMS atau Authenticator App)
   - Verifikasi nomor telepon atau setup authenticator
   - Selesaikan proses aktivasi

**Catatan:** 2-Step Verification **WAJIB** diaktifkan untuk membuat App Password.

---

## 🔑 Langkah 2: Buat App Password

1. Setelah 2-Step Verification aktif, kembali ke [Google Account Security](https://myaccount.google.com/security)
2. Cari bagian **"App passwords"** (atau klik [link ini](https://myaccount.google.com/apppasswords))
3. Jika belum muncul, pastikan 2-Step Verification sudah aktif
4. Klik **"Select app"** dan pilih **"Mail"**
5. Klik **"Select device"** dan pilih **"Other (Custom name)"**
6. Ketik nama aplikasi, misalnya: **"Django Bird App"**
7. Klik **"Generate"**
8. **COPY** App Password yang muncul (16 karakter, tanpa spasi)
   - Format: `xxxx xxxx xxxx xxxx` (copy semua, termasuk spasi atau tanpa spasi)
   - **PENTING:** Password ini hanya muncul sekali, simpan dengan baik!

**Contoh App Password:**

```
abcd efgh ijkl mnop
```

---

## ⚙️ Langkah 3: Konfigurasi di File .env

1. Buka atau buat file `.env` di folder `back/`
2. Tambahkan konfigurasi berikut:

```env
# Email Configuration untuk Gmail
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=abcd efgh ijkl mnop
DEFAULT_FROM_EMAIL=your-email@gmail.com
```

**Ganti:**

- `your-email@gmail.com` → Email Gmail Anda
- `abcd efgh ijkl mnop` → App Password yang sudah Anda copy

**Contoh lengkap:**

```env
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=admin@birddb.org
EMAIL_HOST_PASSWORD=abcd efgh ijkl mnop
DEFAULT_FROM_EMAIL=admin@birddb.org
```

---

## 🧪 Langkah 4: Test Email

### Opsi 1: Test via Django Shell

1. Masuk ke container Docker:

```bash
docker exec -it back-web-1 bash
```

2. Buka Django shell:

```bash
python manage.py shell
```

3. Jalankan kode berikut:

```python
from django.core.mail import send_mail
from django.conf import settings

send_mail(
    subject='Test Email dari Django',
    message='Ini adalah email test. Jika Anda menerima email ini, konfigurasi Gmail sudah benar!',
    from_email=settings.DEFAULT_FROM_EMAIL,
    recipient_list=['pokevhee@gmail.com'],  # Ganti dengan email Anda
    fail_silently=False,
)
```

4. Jika berhasil, Anda akan melihat output:

```
1
```

5. Cek inbox email yang Anda masukkan di `recipient_list`

### Opsi 2: Test via Fitur Forgot Password

1. Jalankan aplikasi Django
2. Buka halaman login di frontend
3. Klik "Lupa Password?"
4. Masukkan email admin yang terdaftar
5. Klik "Kirim Email"
6. Cek inbox email admin

---

## 🔍 Troubleshooting

### Error: "SMTPAuthenticationError: (535, b'5.7.8 Username and Password not accepted')"

**Penyebab:**

- App Password salah atau sudah expired
- Email atau password salah di `.env`

**Solusi:**

1. Pastikan menggunakan **App Password**, bukan password Gmail biasa
2. Generate App Password baru
3. Update `EMAIL_HOST_PASSWORD` di `.env`
4. Restart container Docker

### Error: "SMTPAuthenticationError: (534, b'5.7.9 Application-specific password required')"

**Penyebab:**

- 2-Step Verification belum aktif
- Menggunakan password Gmail biasa, bukan App Password

**Solusi:**

1. Aktifkan 2-Step Verification
2. Buat App Password baru
3. Gunakan App Password di `.env`

### Error: "Connection refused" atau "Timeout"

**Penyebab:**

- Firewall memblokir port 587
- Koneksi internet bermasalah

**Solusi:**

1. Pastikan port 587 tidak diblokir firewall
2. Coba gunakan port alternatif 465 dengan `EMAIL_USE_SSL=True`:

```env
EMAIL_PORT=465
EMAIL_USE_TLS=False
EMAIL_USE_SSL=True
```

### Email tidak terkirim tapi tidak ada error (Return value = 1 tapi email tidak sampai)

**Gejala:**

- `send_mail()` return `1` (berhasil)
- Email tidak sampai ke inbox
- Output email muncul di console dengan format seperti ini:
  ```
  Content-Type: text/plain; charset="utf-8"
  MIME-Version: 1.0
  ...
  ```

**Penyebab:**

- `EMAIL_BACKEND` masih menggunakan **console backend** (default)
- File `.env` belum dibuat atau belum dikonfigurasi
- Container Docker belum di-restart setelah mengubah `.env`

**Solusi:**

1. **Buat atau edit file `.env` di folder `back/`**:

   ```bash
   cd back/
   nano .env  # atau gunakan editor lain
   ```

2. **Pastikan file `.env` berisi**:

   ```env
   EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
   EMAIL_HOST=smtp.gmail.com
   EMAIL_PORT=587
   EMAIL_USE_TLS=True
   EMAIL_HOST_USER=smartml1990@gmail.com
   EMAIL_HOST_PASSWORD=your-app-password-here
   DEFAULT_FROM_EMAIL=smartml1990@gmail.com
   ```

   **PENTING:** Ganti `your-app-password-here` dengan App Password yang sudah Anda buat!

3. **Restart container Docker**:

   ```bash
   docker compose restart back-web-1
   ```

   Atau jika container tidak berjalan:

   ```bash
   docker compose up -d
   ```

4. **Verifikasi konfigurasi** (masuk ke container):

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

   Pastikan:

   - `EMAIL_BACKEND` = `django.core.mail.backends.smtp.EmailBackend`
   - `EMAIL_HOST_PASSWORD` = `SET` (bukan `NOT SET`)

5. **Test kirim email lagi**:

   ```python
   from django.core.mail import send_mail
   from django.conf import settings

   send_mail(
       subject='Test Email dari Django',
       message='Ini adalah email test.',
       from_email=settings.DEFAULT_FROM_EMAIL,
       recipient_list=['pokevhee@gmail.com'],
       fail_silently=False,
   )
   ```

   **Jika menggunakan SMTP dengan benar**, Anda TIDAK akan melihat output email di console. Email akan langsung dikirim via SMTP.

6. **Cek inbox email** (termasuk folder Spam)

---

## 📝 Catatan Penting

1. **App Password hanya muncul sekali** - Simpan dengan baik!
2. **Jangan commit file `.env`** ke Git - Tambahkan ke `.gitignore`
3. **Untuk production**, pertimbangkan menggunakan:
   - SendGrid
   - Mailgun
   - Amazon SES
   - Atau email service provider lainnya
4. **Rate Limiting**: Gmail memiliki batas pengiriman email (sekitar 500 email/hari untuk akun gratis)
5. **Security**: Jangan share App Password dengan siapapun

---

## 🔄 Restart Container Setelah Konfigurasi

Setelah mengubah file `.env`, restart container:

```bash
docker compose restart back-web-1
```

Atau rebuild jika diperlukan:

```bash
docker compose up --build -d
```

---

## ✅ Checklist

- [ ] 2-Step Verification sudah aktif
- [ ] App Password sudah dibuat dan disimpan
- [ ] File `.env` sudah dikonfigurasi dengan benar
- [ ] Container Docker sudah di-restart
- [ ] Test email berhasil terkirim
- [ ] Email tidak masuk ke spam folder

---

## 🧪 Script Test Konfigurasi Email

Gunakan script `test_email_config.py` untuk memverifikasi konfigurasi:

```bash
# Masuk ke container
docker exec -it back-web-1 bash

# Jalankan script test
python manage.py shell < test_email_config.py
```

Script ini akan:

- Menampilkan semua konfigurasi email
- Mengecek apakah menggunakan SMTP atau Console backend
- Mengecek apakah EMAIL_HOST_PASSWORD sudah di-set
- Test koneksi SMTP
- Test kirim email

## 📞 Bantuan Tambahan

Jika masih mengalami masalah:

1. **Cek log Django**:

   ```bash
   docker logs back-web-1
   ```

2. **Test koneksi SMTP manual** (dalam Django shell):

   ```python
   from django.conf import settings
   import smtplib

   server = smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT)
   server.starttls()
   server.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
   print("✅ Koneksi SMTP berhasil!")
   server.quit()
   ```

3. **Pastikan file `.env` ada dan benar**:

   ```bash
   # Cek apakah file .env ada
   ls -la back/.env

   # Lihat isi file .env (pastikan tidak ada spasi di awal/akhir)
   cat back/.env | grep EMAIL
   ```

4. **Pastikan semua environment variables sudah benar**:
   ```bash
   docker exec back-web-1 env | grep EMAIL
   ```

---

**Selamat! Email Gmail Anda sudah siap digunakan! 🎉**
