from django.urls import path
from . import views

app_name = 'personas'

urlpatterns = [
    # Dashboard principal
    path('', views.dashboard, name='dashboard'),

    # Stream MJPEG en vivo (se usa como <img src="...">)
    path('video-feed/', views.video_feed, name='video_feed'),

    # Captura instantánea desde cámara
    path('api/snapshot/', views.deteccion_snapshot, name='snapshot'),

    # Analizar imagen subida por el usuario
    path('api/analizar/', views.analizar_imagen_subida, name='analizar'),

    # Estadísticas para gráfica
    path('api/estadisticas/', views.estadisticas_json, name='estadisticas'),

    # Detener cámara
    path('api/detener/', views.detener_camara, name='detener'),

    # Historial de registros
    path('historial/', views.historial, name='historial'),


    
]


