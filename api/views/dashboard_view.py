from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Count, Q
from django.utils import timezone
from datetime import timedelta

from api.serializers.dashboard_serializer import DashboardStatsSerializer, RecentActivitySerializer
from crud.models import Family, Bird, Image, Sound
from crud.serializers import (
    FamilySerializer, BirdSerializer, ImageSerializer, SoundSerializer,
)


class DashboardViewSet(viewsets.ViewSet):
    """
    ViewSet untuk dashboard statistics dan recent activities
    """

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """
        Endpoint untuk mendapatkan statistik dashboard
        GET /api/dashboard/stats/
        """
        # Hitung total untuk setiap model
        total_families = Family.objects.count()
        total_birds = Bird.objects.count()
        total_images = Image.objects.count()
        total_sounds = Sound.objects.count()

        # Hitung data yang ditambahkan dalam 30 hari terakhir
        thirty_days_ago = timezone.now() - timedelta(days=30)

        recent_families = Family.objects.filter(created_at__gte=thirty_days_ago).count()
        recent_birds = Bird.objects.filter(created_at__gte=thirty_days_ago).count()
        recent_images = Image.objects.filter(created_at__gte=thirty_days_ago).count()
        recent_sounds = Sound.objects.filter(created_at__gte=thirty_days_ago).count()

        # Hitung sound preprocessing status
        preprocessed_sounds = Sound.objects.filter(preprocessing=True).count()
        raw_sounds = Sound.objects.filter(preprocessing=False).count()

        stats_data = {
            'totalFamilies': total_families,
            'totalBirds': total_birds,
            'totalImages': total_images,
            'totalSounds': total_sounds,
            'recentFamilies': recent_families,
            'recentBirds': recent_birds,
            'recentImages': recent_images,
            'recentSounds': recent_sounds,
            'preprocessedSounds': preprocessed_sounds,
            'rawSounds': raw_sounds,
        }

        serializer = DashboardStatsSerializer(stats_data)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def recent_activities(self, request):
        """
        Endpoint untuk mendapatkan aktivitas terbaru
        GET /api/dashboard/recent-activities/
        """
        activities = []

        # Ambil data terbaru dari setiap model (10 terbaru dari masing-masing)
        recent_families = Family.objects.order_by('-created_at')[:10]
        recent_birds = Bird.objects.order_by('-created_at')[:10]
        recent_images = Image.objects.select_related('bird').order_by('-created_at')[:10]
        recent_sounds = Sound.objects.select_related('bird').order_by('-created_at')[:10]

        # Format data family
        for family in recent_families:
            activities.append({
                'id': str(family.id),
                'type': 'family',
                'action': 'created',
                'title': family.family_nm,
                'description': f'New bird family added: {family.description[:100]}...' if len(
                    family.description) > 100 else family.description,
                'timestamp': family.created_at,
                'user': 'System User'  # Bisa diganti dengan user yang sebenarnya jika ada
            })

        # Format data bird
        for bird in recent_birds:
            activities.append({
                'id': str(bird.id),
                'type': 'bird',
                'action': 'created',
                'title': bird.bird_nm,
                'description': f'New bird species added to {bird.family.family_nm} family',
                'timestamp': bird.created_at,
                'user': 'System User'
            })

        # Format data image
        for image in recent_images:
            activities.append({
                'id': str(image.id),
                'type': 'image',
                'action': 'created',
                'title': f'{image.bird.bird_nm} Image',
                'description': f'New image uploaded for {image.bird.bird_nm}',
                'timestamp': image.created_at,
                'user': 'System User'
            })

        # Format data sound
        for sound in recent_sounds:
            activities.append({
                'id': str(sound.id),
                'type': 'sound',
                'action': 'created',
                'title': f'{sound.bird.bird_nm} Recording',
                'description': f'New sound recording from {sound.location}',
                'timestamp': sound.created_at,
                'user': 'System User'
            })

        # Sort by timestamp (terbaru dulu) dan ambil 20 teratas
        activities.sort(key=lambda x: x['timestamp'], reverse=True)
        activities = activities[:20]

        serializer = RecentActivitySerializer(activities, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def monthly_growth(self, request):
        """
        Endpoint untuk mendapatkan data pertumbuhan bulanan
        GET /api/dashboard/monthly-growth/
        """
        # Ambil data 6 bulan terakhir
        now = timezone.now()
        monthly_data = []

        for i in range(5, -1, -1):  # 6 bulan terakhir
            month_start = now.replace(day=1) - timedelta(days=32 * i)
            month_start = month_start.replace(day=1)
            month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)

            families_count = Family.objects.filter(created_at__lte=month_end).count()
            birds_count = Bird.objects.filter(created_at__lte=month_end).count()
            images_count = Image.objects.filter(created_at__lte=month_end).count()
            sounds_count = Sound.objects.filter(created_at__lte=month_end).count()

            monthly_data.append({
                'month': month_start.strftime('%b'),
                'families': families_count,
                'birds': birds_count,
                'images': images_count,
                'sounds': sounds_count
            })

        return Response(monthly_data)

    @action(detail=False, methods=['get'])
    def system_status(self, request):
        """
        Endpoint untuk mendapatkan status sistem
        GET /api/dashboard/system-status/
        """
        try:
            # Test database connection
            Family.objects.count()
            db_status = 'healthy'
        except:
            db_status = 'error'

        # Hitung penggunaan storage (simplified)
        total_images = Image.objects.count()
        total_sounds = Sound.objects.count()
        storage_usage = min(((total_images + total_sounds) / 1000) * 100, 100)  # Simplified calculation

        return Response({
            'database': {
                'status': db_status,
                'message': 'Connected & Healthy' if db_status == 'healthy' else 'Connection Error'
            },
            'storage': {
                'usage_percentage': round(storage_usage),
                'message': f'{round(storage_usage)}% Used'
            },
            'backup': {
                'status': 'warning',
                'message': 'Last: 2 hours ago'
            }
        })


class FamilyViewSet(viewsets.ModelViewSet):
    queryset = Family.objects.all().order_by('-created_at')
    serializer_class = FamilySerializer

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get family statistics"""
        stats = {
            'total': Family.objects.count(),
            'recent': Family.objects.filter(
                created_at__gte=timezone.now() - timedelta(days=30)
            ).count()
        }
        return Response(stats)


class BirdViewSet(viewsets.ModelViewSet):
    queryset = Bird.objects.select_related('family').order_by('-created_at')
    serializer_class = BirdSerializer

    def get_queryset(self):
        queryset = Bird.objects.select_related('family').order_by('-created_at')
        family_id = self.request.query_params.get('family', None)
        if family_id:
            queryset = queryset.filter(family_id=family_id)
        return queryset

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get bird statistics"""
        stats = {
            'total': Bird.objects.count(),
            'recent': Bird.objects.filter(
                created_at__gte=timezone.now() - timedelta(days=30)
            ).count(),
            'by_family': Bird.objects.values('family__family_nm').annotate(
                count=Count('id')
            ).order_by('-count')[:5]
        }
        return Response(stats)


class ImageViewSet(viewsets.ModelViewSet):
    queryset = Image.objects.select_related('bird').order_by('-created_at')
    serializer_class = ImageSerializer

    def get_queryset(self):
        queryset = Image.objects.select_related('bird').order_by('-created_at')
        bird_id = self.request.query_params.get('bird', None)
        if bird_id:
            queryset = queryset.filter(bird_id=bird_id)
        return queryset

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get image statistics"""
        stats = {
            'total': Image.objects.count(),
            'recent': Image.objects.filter(
                created_at__gte=timezone.now() - timedelta(days=30)
            ).count()
        }
        return Response(stats)


class SoundViewSet(viewsets.ModelViewSet):
    queryset = Sound.objects.select_related('bird').order_by('-created_at')
    serializer_class = SoundSerializer

    def get_queryset(self):
        queryset = Sound.objects.select_related('bird').order_by('-created_at')
        bird_id = self.request.query_params.get('bird', None)
        preprocessing = self.request.query_params.get('preprocessing', None)

        if bird_id:
            queryset = queryset.filter(bird_id=bird_id)
        if preprocessing is not None:
            queryset = queryset.filter(preprocessing=preprocessing.lower() == 'true')

        return queryset

    @action(detail=False, methods=['get'])
    def stats(self, request):
        """Get sound statistics"""
        stats = {
            'total': Sound.objects.count(),
            'recent': Sound.objects.filter(
                created_at__gte=timezone.now() - timedelta(days=30)
            ).count(),
            'preprocessed': Sound.objects.filter(preprocessing=True).count(),
            'raw': Sound.objects.filter(preprocessing=False).count()
        }
        return Response(stats)

    @action(detail=True, methods=['post'])
    def toggle_preprocessing(self, request, pk=None):
        """Toggle preprocessing status"""
        sound = self.get_object()
        sound.preprocessing = not sound.preprocessing
        sound.save()
        return Response({'preprocessing': sound.preprocessing})
