from rest_framework import serializers
from ..models import Sound


class SoundSerializer(serializers.ModelSerializer):
    bird_name = serializers.CharField(source='bird.bird_nm', read_only=True)
    sound_url = serializers.SerializerMethodField()

    class Meta:
        model = Sound
        fields = [
            'id', 'bird', 'bird_name', 'sound_file', 'sound_url',
            'recording_date', 'location', 'description', 'preprocessing',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_sound_url(self, obj):
        if obj.sound_file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.sound_file.url)
        return None