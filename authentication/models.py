# Create your models here.
# authentication/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models
import uuid


class User(AbstractUser):
    role = models.CharField(max_length=20, default='user')
    show_password = models.CharField(max_length=40, default='password')
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )