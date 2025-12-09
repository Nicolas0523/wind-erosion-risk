<<<<<<< HEAD
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('get_data/', views.get_data, name='get_data'),
=======
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('get_data/', views.get_data, name='get_data'),
>>>>>>> 39381e6 (Update GEE code for Render)
]