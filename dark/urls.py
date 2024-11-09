"""
URL configuration for dark project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from darknet import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('video_stream_out/', views.video_stream_output, name='video_stream_out'),
    path('video_stream_fea1/', views.video_stream_fea1, name='video_stream_fea1'),
    path('video_stream_fea2/', views.video_stream_fea2, name='video_stream_fea2'),
    path('video_stream_ori/', views.video_stream_ori, name='video_stream_ori'),
    path("control_queues",views.control_queues, name='control_queues')  ,
    path('', views.start_frame_generator_thread, name='start_frame_generator_thread')
]
