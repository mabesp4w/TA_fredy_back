from rest_framework import viewsets, filters, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from ..models import Bird
from ..serializers import BirdSerializer, BirdDetailSerializer
from ..pagination import LaravelStylePagination


class BirdViewSet(viewsets.ModelViewSet):
    queryset = Bird.objects.select_related('family').all()
    serializer_class = BirdSerializer
    pagination_class = LaravelStylePagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['family', 'habitat']
    search_fields = ['bird_nm', 'scientific_nm', 'description', 'habitat']
    ordering_fields = ['bird_nm', 'scientific_nm', 'created_at']
    ordering = ['-created_at']

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return BirdDetailSerializer
        return BirdSerializer

    @action(detail=True, methods=['get'])
    def images(self, request, pk=None):
        """Get all images for this bird"""
        bird = self.get_object()
        images = bird.image_set.all()

        # Apply pagination
        page = self.paginate_queryset(images)
        if page is not None:
            from ..serializers import ImageSerializer
            serializer = ImageSerializer(page, many=True, context={'request': request})
            return self.get_paginated_response(serializer.data)

        from ..serializers import ImageSerializer
        serializer = ImageSerializer(images, many=True, context={'request': request})
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def sounds(self, request, pk=None):
        """Get all sounds for this bird"""
        bird = self.get_object()
        sounds = bird.sound_set.all()

        # Apply pagination
        page = self.paginate_queryset(sounds)
        if page is not None:
            from ..serializers import SoundSerializer
            serializer = SoundSerializer(page, many=True, context={'request': request})
            return self.get_paginated_response(serializer.data)

        from ..serializers import SoundSerializer
        serializer = SoundSerializer(sounds, many=True, context={'request': request})
        return Response(serializer.data)