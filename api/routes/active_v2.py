from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import mediapipe as mp
import numpy as np
import random

from core.vision_math import calculate_head_pose, calculate_blink_ratio
from ml.models_manager import ml_manager 

router = APIRouter()


@router.websocket("/ws/v2/kyc/active-liveness")
async def active_liveness_stream(websocket: WebSocket):
    await websocket.accept()
    
    # Definir el pool de desafíos y elegir 3 aleatorios sin repetir
    all_challenges = ["MIRAR_IZQUIERDA", "MIRAR_DERECHA", "MIRAR_ARRIBA", "PARPADEAR"]
    sequence = random.sample(all_challenges, 3)
    current_step = 0
    
    await websocket.send_json({
        "status": "challenge", 
        "message": f"📍 PASO {current_step+1}/3: Por favor -> {sequence[current_step]}"
    })
    
    frame_count = 0
    try:
        while True:
            data = await websocket.receive_bytes()
            frame_count += 1
            if frame_count % 3 != 0: continue
            
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None: continue
                
            ih, iw, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            results = ml_manager.face_landmarker.detect(mp_image)
            
            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                pitch, yaw, roll = calculate_head_pose(landmarks, iw, ih)
                ear = calculate_blink_ratio(landmarks)
                
                current_challenge = sequence[current_step]
                passed = False
                
                # Evaluar la acción según el desafío actual
                if current_challenge == "MIRAR_IZQUIERDA" and yaw < -35:
                    passed = True
                elif current_challenge == "MIRAR_DERECHA" and yaw > 35:
                    passed = True
                elif current_challenge == "MIRAR_ARRIBA" and pitch > 25:
                    passed = True
                elif current_challenge == "PARPADEAR" and ear < 0.14:
                    passed = True
                    
                if passed:
                    current_step += 1
                    if current_step >= 3:
                        # Si completó los 3, aprueba la sesión
                        await websocket.send_json({
                            "status": "approved", 
                            "message": "✅ ¡Identidad Verificada! Superaste todos los desafíos 3D."
                        })
                        break
                    else:
                        # Si pasó uno, avanza al siguiente
                        await websocket.send_json({
                            "status": "challenge", 
                            "message": f"✅ ¡Bien! Siguiente PASO {current_step+1}/3: -> {sequence[current_step]}"
                        })
                else:
                    # Dar feedback visual de cómo van sus variables
                    metrics = f"Yaw: {yaw:.0f} | Pitch: {pitch:.0f} | Ojos: {ear:.2f}"
                    await websocket.send_json({
                        "status": "pending", 
                        "message": f"Esperando: {current_challenge} ({metrics})"
                    })
            else:
                await websocket.send_json({"status": "warning", "message": "No se detecta rostro."})
                
    except WebSocketDisconnect:
        print("Client disconnected from Active Liveness V2.")