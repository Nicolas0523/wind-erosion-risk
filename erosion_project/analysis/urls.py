from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('get_data/', views.get_data, name='get_data'),
]


urlpatterns = [
    path('', views.home, name='home'),
    path('get_data/', views.get_data, name='get_data'),
]
