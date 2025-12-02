from rest_framework import viewsets, filters, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from ..models import Sound
from ..serializers import SoundSerializer
from ..pagination import LaravelStylePagination


class SoundViewSet(viewsets.ModelViewSet):
    queryset = Sound.objects.select_related('bird').all()
    serializer_class = SoundSerializer
    pagination_class = LaravelStylePagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['bird', ]
    search_fields = ['bird__bird_nm', 'location', 'description']
    ordering_fields = ['created_at']
    ordering = ['-created_at']

    def destroy(self, request, *args, **kwargs):
        """Override destroy to ensure file deletion"""
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(
            {"message": "Sound deleted successfully"},
            status=status.HTTP_204_NO_CONTENT
        )

    @action(detail=False, methods=['get'])
    def preprocessed(self, request):
        """Get all preprocessed sounds"""
        sounds = self.get_queryset()

        # Apply pagination
        page = self.paginate_queryset(sounds)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(sounds, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def unprocessed(self, request):
        """Get all unprocessed sounds"""
        sounds = self.get_queryset()

        # Apply pagination
        page = self.paginate_queryset(sounds)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(sounds, many=True)
        return Response(serializer.data)