# authentication/serializers.py
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework import serializers


User = get_user_model()


class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, style={'input_type': 'password'})
    password_confirm = serializers.CharField(write_only=True, style={'input_type': 'password'})

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password_confirm', 'first_name', 'last_name']
        extra_kwargs = {
            'first_name': {'required': False},
            'last_name': {'required': False}
        }

    def validate(self, data):
        # Validasi password match
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError({"password_confirm": "Password tidak sama"})
        return data

    def create(self, validated_data):
        # Hapus password_confirm dari data
        validated_data.pop('password_confirm')

        # Buat user baru
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email', ''),
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
            role=validated_data.get('role', 'user')  # Default role
        )

        return user


class EmailTokenObtainPairSerializer(TokenObtainPairSerializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['email'] = serializers.EmailField()
        self.fields.pop('username', None)

    def validate(self, attrs):
        # Ambil email dari request
        email = attrs.get('email')

        # Cari user berdasarkan email
        try:
            user = User.objects.get(email=email)
            # Tambahkan username ke attributes untuk diproses TokenObtainPairSerializer
            attrs['username'] = user.username

            # Coba validasi dengan token serializer
            try:
                data = super().validate(attrs)

                # Tambahkan data user ke response
                data['user_id'] = user.id
                data['role'] = user.role

                return data
            except Exception:
                # Password salah, tapi kita berikan pesan yang sama
                raise serializers.ValidationError({'message': 'Kombinasi email dan password salah'})

        except User.DoesNotExist:
            # Email tidak ditemukan, berikan pesan error yang sama
            raise serializers.ValidationError({'message': 'Kombinasi email dan password salah'})
