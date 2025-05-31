from rest_framework import serializers
from ..models import Family, Bird


class FamilySerializer(serializers.ModelSerializer):
    birds_count = serializers.SerializerMethodField()

    class Meta:
        model = Family
        fields = ['id', 'family_nm', 'description', 'birds_count', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_birds_count(self, obj):
        return obj.birds.count()


class FamilyDetailSerializer(serializers.ModelSerializer):
    birds = serializers.SerializerMethodField()

    class Meta:
        model = Family
        fields = ['id', 'family_nm', 'description', 'birds', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_birds(self, obj):
        from .bird_serializer import BirdSerializer
        birds = obj.birds.all()
        return BirdSerializer(birds, many=True).data