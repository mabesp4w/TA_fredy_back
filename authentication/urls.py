from django.urls import path, include

from authentication.views import CustomLogoutView, CustomLoginView, TokenCheckView

urlpatterns = [
    # OAuth2 URLs
    path('login/', CustomLoginView.as_view(), name='costume_login'),
    path('logout/', CustomLogoutView.as_view(), name='custom_logout'),
    path('check-token/', TokenCheckView.as_view(), name='check_token'),
]