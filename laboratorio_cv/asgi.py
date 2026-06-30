import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import personas.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'laboratorio_cv.settings')

application = ProtocolTypeRouter({
    'http': get_asgi_application(),
    'websocket': AuthMiddlewareStack(
        URLRouter(
            personas.routing.websocket_urlpatterns
        )
    ),
})
