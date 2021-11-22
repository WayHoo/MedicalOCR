from django.urls import path
from . import views

urlpatterns = [
    path('upload_img', views.upload_img, name='upload_img'),
    path('login', views.user_login, name='login'),
    path('logout', views.user_logout, name='logout'),
]
