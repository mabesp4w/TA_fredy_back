from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FamilyViewSet, BirdViewSet, ImageViewSet, SoundViewSet

router = DefaultRouter()
router.register(r'families', FamilyViewSet, basename='family')
router.register(r'birds', BirdViewSet, basename='bird')
router.register(r'images', ImageViewSet, basename='image')
router.register(r'sounds', SoundViewSet, basename='sound')

urlpatterns = [
    path('', include(router.urls)),
]