from rest_framework import serializers
from django.db.models import Count, Q
from django.utils import timezone
from datetime import timedelta
from crud.models import Family, Bird, Image, Sound


class FamilySerializer(serializers.ModelSerializer):
    birds_count = serializers.SerializerMethodField()

    class Meta:
        model = Family
        fields = ['id', 'family_nm', 'description', 'birds_count', 'created_at', 'updated_at']

    def get_birds_count(self, obj):
        return obj.birds.count()


class BirdSerializer(serializers.ModelSerializer):
    family_name = serializers.CharField(source='family.family_nm', read_only=True)
    images_count = serializers.SerializerMethodField()
    sounds_count = serializers.SerializerMethodField()

    class Meta:
        model = Bird
        fields = ['id', 'bird_nm', 'scientific_nm', 'family', 'family_name',
                  'description', 'habitat', 'images_count', 'sounds_count',
                  'created_at', 'updated_at']

    def get_images_count(self, obj):
        return obj.image_set.count()

    def get_sounds_count(self, obj):
        return obj.sound_set.count()


class ImageSerializer(serializers.ModelSerializer):
    bird_name = serializers.CharField(source='bird.bird_nm', read_only=True)

    class Meta:
        model = Image
        fields = ['id', 'bird', 'bird_name', 'path_img', 'created_at', 'updated_at']


class SoundSerializer(serializers.ModelSerializer):
    bird_name = serializers.CharField(source='bird.bird_nm', read_only=True)

    class Meta:
        model = Sound
        fields = ['id', 'bird', 'bird_name', 'sound_file', 'recording_date',
                  'location', 'description', 'preprocessing', 'created_at', 'updated_at']


class DashboardStatsSerializer(serializers.Serializer):
    totalFamilies = serializers.IntegerField()
    totalBirds = serializers.IntegerField()
    totalImages = serializers.IntegerField()
    totalSounds = serializers.IntegerField()
    recentFamilies = serializers.IntegerField()
    recentBirds = serializers.IntegerField()
    recentImages = serializers.IntegerField()
    recentSounds = serializers.IntegerField()
    preprocessedSounds = serializers.IntegerField()
    rawSounds = serializers.IntegerField()


class RecentActivitySerializer(serializers.Serializer):
    id = serializers.CharField()
    type = serializers.ChoiceField(choices=['family', 'bird', 'image', 'sound'])
    action = serializers.ChoiceField(choices=['created', 'updated', 'deleted'])
    title = serializers.CharField()
    description = serializers.CharField(required=False)
    timestamp = serializers.DateTimeField()
    user = serializers.CharField(required=False)