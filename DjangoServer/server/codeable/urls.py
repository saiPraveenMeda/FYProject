from django.conf.urls import url
from . import views

app_name = 'codeable'
urlpatterns = [
	url(r'^temp/', views.temp, name = 'temp'),
	url(r'^upload/$', views.FileView, name='file-upload'),
	]