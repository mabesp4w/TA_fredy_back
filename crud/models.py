import os

from django.db import models
import uuid
# Create your models here.

class Family(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    family_nm = models.CharField(max_length=100)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'family'

    def __str__(self):
        return self.family_nm


class Bird(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bird_nm = models.CharField(max_length=100)
    scientific_nm = models.CharField(max_length=100)
    family = models.ForeignKey(Family, on_delete=models.CASCADE, related_name='birds')
    description = models.TextField(null=True, blank=True)
    habitat = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'bird'

    def __str__(self):
        return self.bird_nm

def get_random_filename_images(instance, filename):
    ext = os.path.splitext(filename)[1]  # Mendapatkan ekstensi file (misal: .jpg, .png)
    random_filename = f"{uuid.uuid4().hex}{ext}"  # Membuat nama unik
    bird_nm = instance.bird.bird_nm.strip().lower().replace(' ', '_')
    return os.path.join('images/', bird_nm, random_filename)

class Image(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bird = models.ForeignKey(Bird, on_delete=models.CASCADE)
    path_img = models.ImageField(upload_to=get_random_filename_images, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'image'

    def __str__(self):
        return self.id

    def delete(self, *args, **kwargs):
        # Hapus file path_img jika ada sebelum menghapus data
        if self.path_img and os.path.isfile(self.path_img.path):
            os.remove(self.path_img.path)
        super().delete(*args, **kwargs)


def get_random_filename_sound(instance, filename):
    ext = os.path.splitext(filename)[1]  # Mendapatkan ekstensi file (misal: .jpg, .png)
    random_filename = f"{uuid.uuid4().hex}{ext}"  # Membuat nama unik
    bird_nm = instance.bird.bird_nm.strip().lower().replace(' ', '_')
    return os.path.join('sound/', bird_nm, random_filename)


class Sound(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bird = models.ForeignKey(Bird, on_delete=models.CASCADE)
    sound_file = models.FileField(upload_to=get_random_filename_sound)
    location = models.CharField(max_length=200, null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'sound'

    def __str__(self):
        return f"Sound of {self.bird.id}"

    def delete(self, *args, **kwargs):
        # Hapus file sound_file jika ada sebelum menghapus data
        if self.sound_file and os.path.isfile(self.sound_file.path):
            os.remove(self.sound_file.path)
        super().delete(*args, **kwargs)