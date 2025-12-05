#!/usr/bin/env python
"""
Script untuk test konfigurasi email SMTP
Jalankan: python manage.py shell < test_email_config.py
atau: docker exec back-web-1 python manage.py shell < test_email_config.py
"""

from django.conf import settings
from django.core.mail import send_mail
import smtplib

print("=" * 60)
print("TEST KONFIGURASI EMAIL")
print("=" * 60)

# 1. Cek konfigurasi
print("\n1. KONFIGURASI EMAIL:")
print(f"   EMAIL_BACKEND: {settings.EMAIL_BACKEND}")
print(f"   EMAIL_HOST: {settings.EMAIL_HOST}")
print(f"   EMAIL_PORT: {settings.EMAIL_PORT}")
print(f"   EMAIL_USE_TLS: {settings.EMAIL_USE_TLS}")
print(f"   EMAIL_HOST_USER: {settings.EMAIL_HOST_USER}")
print(f"   EMAIL_HOST_PASSWORD: {'***' if settings.EMAIL_HOST_PASSWORD else 'NOT SET (INI MASALAHNYA!)'}")
print(f"   DEFAULT_FROM_EMAIL: {settings.DEFAULT_FROM_EMAIL}")

# 2. Cek apakah menggunakan SMTP atau Console
print("\n2. STATUS BACKEND:")
if 'console' in settings.EMAIL_BACKEND.lower():
    print("   ⚠️  MASALAH: Masih menggunakan CONSOLE backend!")
    print("   → Email hanya ditampilkan di console, tidak dikirim via SMTP")
    print("   → Solusi: Set EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend di .env")
else:
    print("   ✅ Menggunakan SMTP backend")

# 3. Cek apakah EMAIL_HOST_PASSWORD sudah di-set
print("\n3. CEK PASSWORD:")
if not settings.EMAIL_HOST_PASSWORD:
    print("   ⚠️  MASALAH: EMAIL_HOST_PASSWORD tidak di-set!")
    print("   → Solusi: Tambahkan EMAIL_HOST_PASSWORD di file .env")
    print("   → Gunakan App Password dari Gmail (bukan password biasa)")
else:
    print("   ✅ EMAIL_HOST_PASSWORD sudah di-set")

# 4. Test koneksi SMTP
if 'smtp' in settings.EMAIL_BACKEND.lower() and settings.EMAIL_HOST_PASSWORD:
    print("\n4. TEST KONEKSI SMTP:")
    try:
        server = smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT)
        server.starttls()
        server.login(settings.EMAIL_HOST_USER, settings.EMAIL_HOST_PASSWORD)
        print("   ✅ Koneksi SMTP berhasil!")
        server.quit()
    except Exception as e:
        print(f"   ❌ Error koneksi SMTP: {str(e)}")
        print("   → Periksa App Password dan konfigurasi .env")

# 5. Test kirim email
if 'smtp' in settings.EMAIL_BACKEND.lower() and settings.EMAIL_HOST_PASSWORD:
    print("\n5. TEST KIRIM EMAIL:")
    print("   Mengirim email test...")
    try:
        result = send_mail(
            subject='Test Email dari Django - Konfigurasi SMTP',
            message='Ini adalah email test. Jika Anda menerima email ini, konfigurasi Gmail sudah benar!',
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[settings.EMAIL_HOST_USER],  # Kirim ke email sendiri dulu
            fail_silently=False,
        )
        if result == 1:
            print(f"   ✅ Email berhasil dikirim! (return value: {result})")
            print(f"   → Cek inbox: {settings.EMAIL_HOST_USER}")
            print(f"   → Jangan lupa cek folder SPAM juga!")
        else:
            print(f"   ⚠️  Return value: {result} (seharusnya 1)")
    except Exception as e:
        print(f"   ❌ Error mengirim email: {str(e)}")
        print("   → Periksa log Django untuk detail error")

print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)

