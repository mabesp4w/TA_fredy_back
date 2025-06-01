from django.urls import path, include
from rest_framework.routers import DefaultRouter
from api.views.dashboard_view import FamilyViewSet, BirdViewSet, ImageViewSet, SoundViewSet, DashboardViewSet
from api.views.prediction_view import PredictionViewSet

router = DefaultRouter()
router.register(r'dashboard', DashboardViewSet, basename='dashboard')
router.register(r'families', FamilyViewSet, basename='family')
router.register(r'birds', BirdViewSet, basename='bird')
router.register(r'images', ImageViewSet, basename='image')
router.register(r'sounds', SoundViewSet, basename='sound')

urlpatterns = [
    path('', include(router.urls)),
    path('prediction/', PredictionViewSet.as_view(), name='prediction'),
]

