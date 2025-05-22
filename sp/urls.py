"""
URL configuration for sp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
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
from django.shortcuts import render
from conditionals.views import upload_file, upload_and_encrypt

# Views
def upload_view(request):
    return render(request, 'home.html')

def landing_view(request):
    return render(request, 'landing.html')

def start_view(request):
    return render(request, 'start.html')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', landing_view, name='home'),
    path('file-upload/', upload_view, name='upload'),
    path('start/', start_view, name='start'),
    path('upload/', upload_file, name='upload_file'),
    path('upload_and_encrypt/', upload_and_encrypt, name='upload_and_encrypt'),
]
