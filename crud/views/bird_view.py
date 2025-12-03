from rest_framework import viewsets, filters, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count
from django.utils import timezone
from ..models import Bird
from ..serializers import BirdSerializer, BirdDetailSerializer
from ..pagination import LaravelStylePagination
from api.utils.export_utils import pdf_exporter, excel_exporter


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

    @action(detail=False, methods=['get'])
    def export_pdf(self, request):
        """
        Export semua data birds ke PDF
        GET /crud/birds/export-pdf/
        
        Query parameters (optional):
        - family: Filter by family ID
        - search: Search by bird name, scientific name, description, or habitat
        - ordering: Order by field (default: -created_at)
        """
        try:
            # Get filtered queryset (menggunakan filter yang sama dengan list)
            queryset = self.filter_queryset(self.get_queryset())
            
            # Annotate dengan count images dan sounds
            queryset = queryset.annotate(
                image_count=Count('image'),
                sound_count=Count('sound')
            )
            
            # Prepare data dengan semua field
            data = []
            for bird in queryset:
                # Truncate description dan habitat jika terlalu panjang untuk PDF
                description = (bird.description[:150] + '...') if bird.description and len(bird.description) > 150 else (bird.description or '-')
                habitat = (bird.habitat[:150] + '...') if bird.habitat and len(bird.habitat) > 150 else (bird.habitat or '-')
                
                data.append({
                    'Bird Name': bird.bird_nm,
                    'Scientific Name': bird.scientific_nm,
                    'Family': bird.family.family_nm if bird.family else '-',
                    'Description': description,
                    'Habitat': habitat,
                    'Images Count': bird.image_count,
                    'Sounds Count': bird.sound_count,
                    'Created At': bird.created_at.strftime('%Y-%m-%d') if bird.created_at else '-'
                })
            
            headers = [
                'Bird Name',
                'Scientific Name',
                'Family',
                'Description',
                'Habitat',
                'Images Count',
                'Sounds Count',
                'Created At'
            ]
            
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            filename = f"birds_export_{timestamp}.pdf"
            
            return pdf_exporter.export_to_pdf(
                data=data,
                filename=filename,
                title="Laporan Data Burung",
                headers=headers
            )
        except Exception as e:
            return Response(
                {'error': f'Failed to generate PDF: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['get'])
    def export_excel(self, request):
        """
        Export semua data birds ke Excel
        GET /crud/birds/export-excel/
        
        Query parameters (optional):
        - family: Filter by family ID
        - search: Search by bird name, scientific name, description, or habitat
        - ordering: Order by field (default: -created_at)
        """
        try:
            # Get filtered queryset (menggunakan filter yang sama dengan list)
            queryset = self.filter_queryset(self.get_queryset())
            
            # Annotate dengan count images dan sounds
            queryset = queryset.annotate(
                image_count=Count('image'),
                sound_count=Count('sound')
            )
            
            # Prepare data dengan semua field (tanpa truncate untuk Excel)
            data = []
            for bird in queryset:
                data.append({
                    'Bird Name': bird.bird_nm,
                    'Scientific Name': bird.scientific_nm,
                    'Family': bird.family.family_nm if bird.family else '-',
                    'Description': bird.description or '-',
                    'Habitat': bird.habitat or '-',
                    'Images Count': bird.image_count,
                    'Sounds Count': bird.sound_count,
                    'Created At': bird.created_at.strftime('%Y-%m-%d %H:%M:%S') if bird.created_at else '-',
                    'Updated At': bird.updated_at.strftime('%Y-%m-%d %H:%M:%S') if bird.updated_at else '-'
                })
            
            headers = [
                'Bird Name',
                'Scientific Name',
                'Family',
                'Description',
                'Habitat',
                'Images Count',
                'Sounds Count',
                'Created At',
                'Updated At'
            ]
            
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            filename = f"birds_export_{timestamp}.xlsx"
            
            return excel_exporter.export_to_excel(
                data=data,
                filename=filename,
                title="Laporan Data Burung",
                headers=headers,
                sheet_name="Birds Data"
            )
        except Exception as e:
            return Response(
                {'error': f'Failed to generate Excel: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )