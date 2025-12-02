from rest_framework import viewsets, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from ..models import Family
from ..serializers import FamilySerializer, FamilyDetailSerializer
from ..pagination import LaravelStylePagination


class FamilyViewSet(viewsets.ModelViewSet):
    queryset = Family.objects.all()
    serializer_class = FamilySerializer
    pagination_class = LaravelStylePagination
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['family_nm', 'description']
    ordering_fields = ['family_nm', 'created_at']
    ordering = ['-created_at']

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return FamilyDetailSerializer
        return FamilySerializer

    @action(detail=True, methods=['get'])
    def birds(self, request, pk=None):
        """Get all birds in this family"""
        family = self.get_object()
        birds = family.birds.all()

        # Apply pagination
        page = self.paginate_queryset(birds)
        if page is not None:
            from ..serializers import BirdSerializer
            serializer = BirdSerializer(page, many=True, context={'request': request})
            return self.get_paginated_response(serializer.data)

        from ..serializers import BirdSerializer
        serializer = BirdSerializer(birds, many=True, context={'request': request})
        return Response(serializer.data)