from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from ml.models_manager import ml_manager

CONFIDENCE_THRESHOLD = 0.85

router = APIRouter()


@router.websocket("/ws/v1/kyc/liveness-stream")
async def liveness_stream(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0 # Counter to manage frame sampling
    
    try:
        while True:
            # Receive raw bytes from the client
            data = await websocket.receive_bytes()
            frame_count += 1
            
            # Traffic control: Only process every 5th frame (approx. 6 FPS)
            if frame_count % 5 != 0:
                continue
                
            # Decode the byte stream into an OpenCV image
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Phase 1: Face Detection (Gatekeeper)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = ml_manager.face_detector.detect(mp_image)

            if not results.detections:
                await websocket.send_json({"status": "error", "message": "No face detected"})
                continue
            
            # Business Rule: Ensure only one person is in the frame
            if len(results.detections) > 1:
                await websocket.send_json({"status": "error", "message": "Multiple faces detected"})
                continue
            
            # TODO: Integrate Phase 2 (PyTorch Liveness Model) here
            await websocket.send_json({"status": "processing", "message": "Face isolated, running AI check..."})

            # Strict preprocessing pipeline for inference
            inference_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # 1. Cropping ✂️
            # Extract absolute bounding box coordinates from modern MediaPipe API
            bbox = results.detections[0].bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            
            # Extract total image dimensions
            ih, iw, _ = frame.shape
            
            # Ensure the starting coordinates do not fall into negative pixels
            x, y = max(0, x), max(0, y)
            
            # Secure the ending coordinates using min() to avoid index out of bounds
            x_end = min(iw, x+w)
            y_end = min(ih, y+h)
            
            # Perform the NumPy array slicing to isolate the face
            face_crop = frame[y:y_end, x:x_end]
            
            if face_crop.size == 0:
                continue

            # 2. Transformation 📐
            face_tensor = inference_transforms(face_crop)
            # Add a batch dimension to match model's expected input shape: [1, 3, 224, 224]
            face_tensor = face_tensor.unsqueeze(0).to(ml_manager.device)

            # 3. Inference 🧠
            with torch.no_grad():
                outputs = ml_manager.liveness_model(face_tensor)
    
                # Apply Softmax to get readable probabilities
                probabilities = F.softmax(outputs, dim=1)
                
                # Extract the float values for both classes
                # Index 0: Live, Index 1: Spoof
                prob_live = probabilities[0][0].item()
                prob_spoof = probabilities[0][1].item()

            if prob_live >= CONFIDENCE_THRESHOLD:
                await websocket.send_json({
                    "status": "success",
                    "message": "Identity Verified",
                    "confidence": round(prob_live, 4)
                })
            else:
                # If the confidence is too low, or if spoofing is directly detected
                await websocket.send_json({
                    "status": "error",
                    "message": "Spoofing Detected or Confidence too low",
                    "confidence": round(prob_live, 4)
                })

    except WebSocketDisconnect:
        print("🔌 Client disconnected from stream.")