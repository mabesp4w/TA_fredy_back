# myapp/backends.py
from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend

User = get_user_model()

class EmailBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        # Tambahkan pengecekan untuk username yang None
        if username is None:
            return None

        # Jika username berformat email
        if '@' in username:
            try:
                user = User.objects.get(email=username)
                if user.check_password(password):
                    return user
            except User.DoesNotExist:
                return None
        # Jika tidak, gunakan authenticate bawaan
        return super().authenticate(request, username=username, password=password, **kwargs)