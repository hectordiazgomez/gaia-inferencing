from django.urls import path
from . import views

urlpatterns = [
    path('translate/', views.translation_endpoint, name='translation_endpoint'),
    path('get-details/', views.get_details, name='get_method'),
]