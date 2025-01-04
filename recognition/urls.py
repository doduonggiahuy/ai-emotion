from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.yolo_detect, name='yolo_detect'),
    path('video_feed/', views.video_feed, name='video_feed'),
]
