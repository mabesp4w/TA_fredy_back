from rest_framework import serializers
from ..models import Image


class ImageSerializer(serializers.ModelSerializer):
    bird_name = serializers.CharField(source='bird.bird_nm', read_only=True)
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = Image
        fields = ['id', 'bird', 'bird_name', 'path_img', 'image_url', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']

    def get_image_url(self, obj):
        if obj.path_img:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.path_img.url)
        return None