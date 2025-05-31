from datetime import timedelta

from django.contrib.auth import authenticate
from django.utils import timezone
from oauth2_provider.contrib.rest_framework import TokenHasScope
from oauth2_provider.settings import oauth2_settings
from oauthlib.common import generate_token
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from oauth2_provider.models import AccessToken, RefreshToken

from oauth2_provider.models import Application

from authentication.serializers import UserRegistrationSerializer
from thobias import settings


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
