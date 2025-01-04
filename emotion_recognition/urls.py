from django.contrib import admin
from django.urls import path
from recognition import views  # Import views từ ứng dụng recognition

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),  # Thêm đường dẫn tới view 'index'
    path('video_feed/', views.video_feed, name='video_feed'),  # Thêm đường dẫn tới video stream
]
