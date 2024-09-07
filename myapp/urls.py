from django.urls import path
from . import views

urlpatterns = [
    path('translate/', views.translation_endpoint, name='translation_endpoint'),
]