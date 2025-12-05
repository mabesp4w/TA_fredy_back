from datetime import timedelta

from django.conf import settings
from django.contrib.auth import authenticate, get_user_model
from django.utils import timezone
from django.core.mail import send_mail
from oauth2_provider.contrib.rest_framework import TokenHasScope
from oauth2_provider.settings import oauth2_settings
from oauthlib.common import generate_token
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from oauth2_provider.models import AccessToken, RefreshToken

from oauth2_provider.models import Application

User = get_user_model()


class AdminOnlyView(APIView):
    permission_classes = [TokenHasScope]
    required_scopes = ['admin']

    def get(self, request):
        return Response({"message": "Admin access granted"})


class UserView(APIView):
    permission_classes = [TokenHasScope]
    required_scopes = ['user']

    def get(self, request):
        return Response({"message": "User access granted"})

# login
class CustomLoginView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):

        # Coba ambil data dari berbagai sumber
        username = request.data.get('username') or request.POST.get('username')
        password = request.data.get('password') or request.POST.get('password')

        # Validasi parameter yang diperlukan
        if not all([username, password]):
            return Response({"detail": "Username dan password diperlukan"},
                            status=status.HTTP_400_BAD_REQUEST)

        # Ambil client_id dan client_secret dari settings
        client_id = settings.OAUTH2_CLIENT_ID
        client_secret = settings.OAUTH2_CLIENT_SECRET

        # Validasi client
        try:
            application = Application.objects.get(client_id=client_id, client_secret=client_secret)
        except Application.DoesNotExist:
            return Response({"detail": "Konfigurasi OAuth tidak valid di server"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Autentikasi user (bisa dengan email atau username)
        if '@' in username:
            # Jika input berupa email, cari user berdasarkan email
            from django.contrib.auth import get_user_model
            User = get_user_model()
            try:
                user_obj = User.objects.get(email=username)
                user = authenticate(username=user_obj.username, password=password)
            except User.DoesNotExist:
                user = None
        else:
            # Autentikasi normal dengan username
            user = authenticate(username=username, password=password)

        if not user:
            return Response({"detail": "Username/email atau password salah"},
                            status=status.HTTP_401_UNAUTHORIZED)

        # Hapus token lama untuk user ini jika ada
        # AccessToken.objects.filter(user=user, application=application).delete()
        # RefreshToken.objects.filter(user=user, application=application).delete()

        # Buat token baru
        expires = timezone.now() + timedelta(seconds=oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS)
        access_token = AccessToken.objects.create(
            user=user,
            application=application,
            token=generate_token(),
            expires=expires,
            scope='read write'
        )

        # Buat refresh token
        refresh_token = RefreshToken.objects.create(
            user=user,
            application=application,
            token=generate_token(),
            access_token=access_token
        )

        # Buat response
        response = {
            "access_token": access_token.token,
            "expires_in": oauth2_settings.ACCESS_TOKEN_EXPIRE_SECONDS,
            "token_type": "Bearer",
            "scope": access_token.scope,
            "refresh_token": refresh_token.token,
            "user": {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": getattr(user, 'role', 'user'),
                "first_name": getattr(user, 'first_name', user.first_name),
            }
        }

        return Response(response)


class TokenCheckView(APIView):
    authentication_classes = []  # Kosongkan autentikasi untuk pengecekan token
    permission_classes = []  # Kosongkan permission untuk pengecekan token

    def post(self, request):
        # Coba ambil token dari header Authorization
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')

        # Logging untuk debug
        print("Authorization header:", auth_header)

        if not auth_header.startswith('Bearer '):
            return Response({
                "valid": False,
                "detail": "Header Authorization tidak valid atau tidak ditemukan. Format: 'Bearer {token}'"
            }, status=status.HTTP_400_BAD_REQUEST)

        # Ekstrak token dari header
        token = auth_header.split(' ')[1].strip()

        # Jika masih tidak ada token
        if not token:
            return Response({
                "valid": False,
                "detail": "Token tidak disediakan"
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            access_token = AccessToken.objects.get(token=token)

            # Cek apakah token sudah kedaluwarsa
            if access_token.expires < timezone.now():
                return Response({
                    "valid": False,
                    "detail": "Token sudah kedaluwarsa"
                }, status=status.HTTP_401_UNAUTHORIZED)

            # Token valid
            return Response({
                "valid": True,
                "user_id": str(access_token.user.id),
                "username": access_token.user.username,
                "email": access_token.user.email,
                "role": access_token.user.role,
                "scope": access_token.scope,
                "expires": access_token.expires
            })

        except AccessToken.DoesNotExist:
            return Response({
                "valid": False,
                "detail": "Token tidak valid atau tidak ditemukan"
            }, status=status.HTTP_401_UNAUTHORIZED)


class CustomLogoutView(APIView):
    def post(self, request):
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        if not auth_header.startswith('Bearer '):
            return Response({"detail": "Header Authorization tidak valid"},
                            status=status.HTTP_400_BAD_REQUEST)

        token = auth_header.split(' ')[1]

        try:
            access_token = AccessToken.objects.get(token=token)
            access_token.delete()
            return Response({"detail": "Logout berhasil, token telah dihapus"},
                            status=status.HTTP_200_OK)
        except AccessToken.DoesNotExist:
            return Response({"detail": "Token tidak valid atau tidak ditemukan"},
                            status=status.HTTP_400_BAD_REQUEST)


class ForgotPasswordView(APIView):
    """
    Endpoint untuk lupa password
    Mengirim email berisi informasi password admin
    """
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        email = request.data.get('email') or request.POST.get('email')

        if not email:
            return Response(
                {"detail": "Email harus diisi"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Cari admin user berdasarkan email
        try:
            admin_user = User.objects.filter(
                email=email,
                role='admin',
                is_staff=True
            ).first()

            if not admin_user:
                # Untuk keamanan, tetap return success meskipun email tidak ditemukan
                # Ini mencegah email enumeration attack
                return Response(
                    {
                        "detail": "Email tidak terdaftar sebagai admin"
                    },
                    status=status.HTTP_200_OK
                )

            # Ambil password dari show_password field
            # Jika show_password kosong atau tidak ada atau masih default, gunakan default message
            admin_password = admin_user.show_password if (hasattr(admin_user, 'show_password') and admin_user.show_password and admin_user.show_password.strip() and admin_user.show_password != 'password') else 'Silakan hubungi administrator untuk mendapatkan password'

            # Kirim email
            subject = 'Informasi Password Admin - Sistem Identifikasi Burung'
            message = f"""
Halo,

Anda telah meminta informasi password untuk akun admin Anda.

Informasi Login:
- Email: {admin_user.email}
- Password: {admin_password}

Silakan gunakan informasi di atas untuk login ke sistem.

Jika Anda tidak meminta informasi ini, abaikan email ini.

Terima kasih,
Sistem Identifikasi Burung
            """
            from_email = settings.DEFAULT_FROM_EMAIL
            recipient_list = [email]

            # Log konfigurasi email untuk debugging
            import logging
            logger = logging.getLogger(__name__)
            
            # Print ke console untuk debugging langsung
            print("=" * 60)
            print("FORGOT PASSWORD - EMAIL CONFIGURATION")
            print("=" * 60)
            print(f"EMAIL_BACKEND: {settings.EMAIL_BACKEND}")
            print(f"EMAIL_HOST: {settings.EMAIL_HOST}")
            print(f"EMAIL_HOST_USER: {settings.EMAIL_HOST_USER}")
            print(f"EMAIL_HOST_PASSWORD: {'SET' if settings.EMAIL_HOST_PASSWORD else 'NOT SET (MASALAH!)'}")
            print(f"EMAIL_PORT: {settings.EMAIL_PORT}")
            print(f"EMAIL_USE_TLS: {settings.EMAIL_USE_TLS}")
            print(f"DEFAULT_FROM_EMAIL: {settings.DEFAULT_FROM_EMAIL}")
            print(f"Recipient: {email}")
            print("=" * 60)
            
            logger.info(f"Attempting to send email to: {email}")
            logger.info(f"EMAIL_BACKEND: {settings.EMAIL_BACKEND}")
            logger.info(f"EMAIL_HOST: {settings.EMAIL_HOST}")
            logger.info(f"EMAIL_HOST_USER: {settings.EMAIL_HOST_USER}")
            logger.info(f"EMAIL_HOST_PASSWORD: {'SET' if settings.EMAIL_HOST_PASSWORD else 'NOT SET'}")

            # Cek apakah masih menggunakan console backend
            if 'console' in settings.EMAIL_BACKEND.lower():
                error_msg = "EMAIL_BACKEND masih menggunakan console backend! Email tidak akan terkirim via SMTP. Pastikan file .env sudah dikonfigurasi dan container sudah di-restart."
                print(f"ERROR: {error_msg}")
                logger.error(error_msg)
                return Response(
                    {
                        "detail": "Konfigurasi email belum benar. Silakan hubungi administrator."
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Cek apakah EMAIL_HOST_PASSWORD sudah di-set
            if not settings.EMAIL_HOST_PASSWORD:
                error_msg = "EMAIL_HOST_PASSWORD tidak di-set! Pastikan file .env sudah dikonfigurasi dengan App Password."
                print(f"ERROR: {error_msg}")
                logger.error(error_msg)
                return Response(
                    {
                        "detail": "Konfigurasi email belum lengkap. Silakan hubungi administrator."
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            try:
                result = send_mail(
                    subject,
                    message,
                    from_email,
                    recipient_list,
                    fail_silently=False,
                )
                print(f"Email send result: {result}")
                logger.info(f"Email send result: {result}")
                
                if result == 1:
                    print(f"✅ Email successfully sent to {email}")
                    logger.info(f"Email successfully sent to {email}")
                    return Response(
                        {
                            "detail": "Email berhasil dikirim. Silakan cek inbox email Anda."
                        },
                        status=status.HTTP_200_OK
                    )
                else:
                    warning_msg = f"Email send returned unexpected value: {result}"
                    print(f"WARNING: {warning_msg}")
                    logger.warning(warning_msg)
                    return Response(
                        {
                            "detail": "Email mungkin tidak terkirim. Silakan coba lagi atau hubungi administrator."
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            except Exception as e:
                # Log error untuk debugging
                error_msg = f"Error sending email: {str(e)}"
                print(f"ERROR: {error_msg}")
                logger.error(error_msg, exc_info=True)
                
                return Response(
                    {
                        "detail": f"Gagal mengirim email: {str(e)}. Silakan hubungi administrator."
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in forgot password: {str(e)}")
            
            return Response(
                {
                    "detail": "Terjadi kesalahan. Silakan coba lagi nanti."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
