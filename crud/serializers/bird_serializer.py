from rest_framework import serializers
from ..models import Bird, Image, Sound


class BirdSerializer(serializers.ModelSerializer):
    family_name = serializers.CharField(source='family.family_nm', read_only=True)
    images_count = serializers.SerializerMethodField()
    sounds_count = serializers.SerializerMethodField()

    class Meta:
        model = Bird
        fields = [
            'id', 'bird_nm', 'scientific_nm', 'family', 'family_name',
            'description', 'habitat', 'images_count', 'sounds_count',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_images_count(self, obj):
        return obj.image_set.count()

    def get_sounds_count(self, obj):
        return obj.sound_set.count()


class BirdDetailSerializer(serializers.ModelSerializer):
    family = serializers.SerializerMethodField()
    images = serializers.SerializerMethodField()
    sounds = serializers.SerializerMethodField()

    class Meta:
        model = Bird
        fields = [
            'id', 'bird_nm', 'scientific_nm', 'family',
            'description', 'habitat', 'images', 'sounds',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_family(self, obj):
        from .family_serializer import FamilySerializer
        return FamilySerializer(obj.family).data

    def get_images(self, obj):
        from .image_serializer import ImageSerializer
        images = obj.image_set.all()
        return ImageSerializer(images, many=True).data

    def get_sounds(self, obj):
        from .sound_serializer import SoundSerializer
        sounds = obj.sound_set.all()
        return SoundSerializer(sounds, many=True).data