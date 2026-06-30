"""
views.py — Conteo de personas con OpenCV en Django
Detecta personas de pie (YOLO) y personas sentadas (YOLO + Haar rostro como respaldo).
"""

import cv2
import numpy as np
import base64
import os
import threading
from datetime import datetime

from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from django.conf import settings
from django.core.files.base import ContentFile

from PIL import Image

from .models import RegistroConteo
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────

USE_YOLO = True
yolo_model = None

try:
    from ultralytics import YOLO
    _modelo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'yolov8n.pt'
    )
    yolo_model = YOLO(_modelo_path)
    USE_YOLO = True
    print("[CV] YOLOv8 cargado correctamente.", flush=True)
except Exception as e:
    print(f"[CV] YOLO falló: {e}", flush=True)
    USE_YOLO = False

# HOG — fallback si YOLO no está disponible
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Haar rostro — respaldo para personas sentadas
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Estado global de la cámara
camara_lock = threading.Lock()
camara_activa = False
cap_global = None


# ─────────────────────────────────────────────
#  DETECCIÓN
# ─────────────────────────────────────────────

def detectar_personas_hog(frame):
    frame_anotado = frame.copy()
    personas = []

    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(16, 16),
        padding=(8, 8),
        scale=1.03,
        hitThreshold=-0.5,
        useMeanshiftGrouping=False
    )

    if len(boxes) > 0:
        weights_list = [float(w[0]) if hasattr(w, '__len__') else float(w) for w in weights]
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), weights_list, 0.0, 0.3)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                personas.append((int(x), int(y), int(w), int(h)))

    # Haar rostro como respaldo
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    for (fx, fy, fw, fh) in rostros:
        solapado = any(
            fx < px + pw and fx + fw > px and fy < py + ph and fy + fh > py
            for (px, py, pw, ph) in personas
        )
        if not solapado:
            personas.append((fx, fy, fw, fh))

    for idx, (x, y, w, h) in enumerate(personas):
        cv2.rectangle(frame_anotado, (x, y), (x + w, y + h), (0, 200, 80), 2)
        label = f'P{idx + 1}'
        cv2.rectangle(frame_anotado, (x, y - 24), (x + 40, y), (0, 200, 80), -1)
        cv2.putText(frame_anotado, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    _dibujar_overlay(frame_anotado, len(personas))
    return frame_anotado, len(personas), personas


def detectar_personas_yolo(frame):
    frame_anotado = frame.copy()
    personas = []

    # YOLO — cuerpo completo y medio cuerpo
    resultados = yolo_model(frame, classes=[0], conf=0.15, imgsz=640, verbose=False)
    for r in resultados:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            personas.append((x1, y1, x2 - x1, y2 - y1))
            cv2.rectangle(frame_anotado, (x1, y1), (x2, y2), (0, 200, 80), 2)
            label = f'P{len(personas)} {conf:.0%}'
            cv2.rectangle(frame_anotado, (x1, y1 - 24), (x1 + 80, y1), (0, 200, 80), -1)
            cv2.putText(frame_anotado, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Haar rostro — respaldo para personas sentadas que YOLO no detectó
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    for (fx, fy, fw, fh) in rostros:
        solapado = any(
            fx < px + pw and fx + fw > px and fy < py + ph and fy + fh > py
            for (px, py, pw, ph) in personas
        )
        if not solapado:
            personas.append((fx, fy, fw, fh))
            cv2.rectangle(frame_anotado, (fx, fy), (fx + fw, fy + fh), (255, 140, 0), 2)
            label = f'P{len(personas)} (rostro)'
            cv2.rectangle(frame_anotado, (fx, fy - 24), (fx + 100, fy), (255, 140, 0), -1)
            cv2.putText(frame_anotado, label, (fx + 4, fy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    _dibujar_overlay(frame_anotado, len(personas))
    return frame_anotado, len(personas), personas


def detectar_personas(frame):
    if USE_YOLO and yolo_model is not None:
        return detectar_personas_yolo(frame)
    return detectar_personas_hog(frame)


def _dibujar_overlay(frame, total):
    color = (0, 180, 80) if total < 10 else (0, 120, 255)
    cv2.rectangle(frame, (10, 10), (280, 55), (20, 20, 20), -1)
    modo = 'YOLO+Haar' if USE_YOLO else 'HOG+Haar'
    cv2.putText(frame,
                f'Personas: {total}  [{modo}]',
                (16, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
    ts = datetime.now().strftime('%d/%m/%Y  %H:%M:%S')
    cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────

def frame_a_jpeg(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buffer.tobytes()


def frame_a_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')


# ─────────────────────────────────────────────
#  STREAM
# ─────────────────────────────────────────────

def generar_stream_frames():
    global cap_global, camara_activa

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)

    with camara_lock:
        cap_global = cap
        camara_activa = True

    frame_count = 0
    ultimo_frame_anotado = None

    try:
        while camara_activa:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 == 0 or ultimo_frame_anotado is None:
                ultimo_frame_anotado, total, _ = detectar_personas(frame)

            jpeg = frame_a_jpeg(ultimo_frame_anotado)
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            )
    finally:
        cap.release()
        with camara_lock:
            cap_global = None
            camara_activa = False


# ─────────────────────────────────────────────
#  VISTAS DJANGO
# ─────────────────────────────────────────────

def dashboard(request):
    ultimos_registros = RegistroConteo.objects.all()[:10]
    context = {
        'ultimos_registros': ultimos_registros,
        'total_registros': RegistroConteo.objects.count(),
    }
    return render(request, 'personas/dashboard.html', context)


def video_feed(request):
    return StreamingHttpResponse(
        generar_stream_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@require_GET
def deteccion_snapshot(request):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JsonResponse({'error': 'No se pudo acceder a la cámara.'}, status=500)

    frame_anotado, total, personas = detectar_personas(frame)
    img_b64 = frame_a_base64(frame_anotado)
    img_bytes = base64.b64decode(img_b64)
    nombre_archivo = f'snap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'

    registro = RegistroConteo(cantidad_personas=total, fuente='camara')
    registro.captura.save(nombre_archivo, ContentFile(img_bytes), save=True)

    return JsonResponse({
        'cantidad': total,
        'timestamp': registro.timestamp.strftime('%d/%m/%Y %H:%M:%S'),
        'imagen_b64': img_b64,
        'registro_id': registro.id,
    })


@csrf_exempt
@require_POST
def analizar_imagen_subida(request):
    if 'imagen' not in request.FILES:
        return JsonResponse({'error': 'No se recibió ninguna imagen.'}, status=400)

    archivo = request.FILES['imagen']
    imagen_pil = Image.open(archivo).convert('RGB')
    frame = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)

    frame_anotado, total, personas = detectar_personas(frame)
    img_b64 = frame_a_base64(frame_anotado)
    img_bytes = base64.b64decode(img_b64)
    nombre_archivo = f'upload_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'

    registro = RegistroConteo(cantidad_personas=total, fuente='imagen')
    registro.captura.save(nombre_archivo, ContentFile(img_bytes), save=True)

    return JsonResponse({
        'cantidad': total,
        'timestamp': registro.timestamp.strftime('%d/%m/%Y %H:%M:%S'),
        'imagen_b64': img_b64,
        'registro_id': registro.id,
        'rectangulos': [{'x': x, 'y': y, 'w': w, 'h': h} for (x, y, w, h) in personas],
    })


@require_GET
def historial(request):
    registros = RegistroConteo.objects.all()
    return render(request, 'personas/historial.html', {'registros': registros})


@require_GET
def estadisticas_json(request):
    registros = RegistroConteo.objects.all()[:20]
    data = [
        {
            'timestamp': r.timestamp.strftime('%H:%M:%S'),
            'cantidad': r.cantidad_personas,
            'fuente': r.fuente,
        }
        for r in reversed(list(registros))
    ]
    promedio = sum(r['cantidad'] for r in data) / len(data) if data else 0
    maximo = max((r['cantidad'] for r in data), default=0)

    return JsonResponse({
        'registros': data,
        'promedio': round(promedio, 1),
        'maximo': maximo,
        'total_snapshots': RegistroConteo.objects.count(),
    })


@require_GET
def detener_camara(request):
    global camara_activa
    with camara_lock:
        camara_activa = False
    return JsonResponse({'status': 'camara_detenida'})

