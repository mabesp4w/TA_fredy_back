from rest_framework import viewsets, filters, status
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from ..models import Image
from ..serializers import ImageSerializer
from ..pagination import LaravelStylePagination


class ImageViewSet(viewsets.ModelViewSet):
    queryset = Image.objects.select_related('bird').all()
    serializer_class = ImageSerializer
    pagination_class = LaravelStylePagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['bird']
    search_fields = ['bird__bird_nm']
    ordering_fields = ['created_at']
    ordering = ['-created_at']

    def destroy(self, request, *args, **kwargs):
        """Override destroy to ensure file deletion"""
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(
            {"message": "Image deleted successfully"},
            status=status.HTTP_204_NO_CONTENT
        )