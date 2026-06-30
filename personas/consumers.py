"""
consumers.py — WebSocket para enviar conteos en tiempo real.
"""
import json
import asyncio
import cv2
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
from .views import detectar_personas_hog, frame_a_base64


class ConteoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.streaming = True
        asyncio.ensure_future(self.send_frames())

    async def disconnect(self, close_code):
        self.streaming = False

    async def send_frames(self):
        cap = cv2.VideoCapture(0)
        try:
            while self.streaming:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_anotado, total, _ = detectar_personas_hog(frame)
                img_b64 = frame_a_base64(frame_anotado)
                await self.send(text_data=json.dumps({
                    'tipo': 'frame',
                    'imagen': img_b64,
                    'cantidad': total,
                }))
                await asyncio.sleep(0.1)  # ~10 fps via WebSocket
        finally:
            cap.release()
