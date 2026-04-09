import cv2
import numpy as np
import math
import mediapipe as mp
import torch
from torchvision import transforms


def calculate_head_pose(landmarks, img_w, img_h):
    """
    Calculates the 3D head pose (Pitch, Yaw, Roll) using a mathematically
    aligned 3D face model against the camera's coordinate system.
    """
    # 1. 2D Image Points (Camera Space: +X is Right, +Y is Down)
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),     # Nose tip
        (landmarks[152].x * img_w, landmarks[152].y * img_h), # Chin
        (landmarks[33].x * img_w, landmarks[33].y * img_h),   # Left eye
        (landmarks[263].x * img_w, landmarks[263].y * img_h), # Right eye
        (landmarks[61].x * img_w, landmarks[61].y * img_h),   # Left mouth
        (landmarks[291].x * img_w, landmarks[291].y * img_h)  # Right mouth
    ], dtype="double")

    # 2. 3D Model Points (ALIGNED TO CAMERA SPACE)
    # Y is inverted (Down is positive)
    # Z is inverted (Away from camera is positive)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, 330.0, 65.0),          # Chin
        (-225.0, -170.0, 135.0),     # Left eye corner
        (225.0, -170.0, 135.0),      # Right eye corner
        (-150.0, 150.0, 125.0),      # Left mouth corner
        (150.0, 150.0, 125.0)        # Right mouth corner
    ])

    # 3. Camera Internals Simulation
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))

    # 4. Solve Perspective-n-Point
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    # 5. Intuitive Standardization
    # Convert OpenCV camera rotations to human-intuitive directions:
    # UP = Positive Pitch, DOWN = Negative Pitch
    # RIGHT = Positive Yaw, LEFT = Negative Yaw
    pitch = -angles[0] 
    yaw = -angles[1]
    roll = angles[2]

    return pitch, yaw, roll


def calculate_blink_ratio(landmarks):
    # Función para calcular la distancia euclidiana entre dos puntos
    def get_distance(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    # Ojo Izquierdo (Puntos de MediaPipe: 159 arriba, 145 abajo, 33 izq, 133 der)
    le_v = get_distance(landmarks[159], landmarks[145])
    le_h = get_distance(landmarks[33], landmarks[133])
    ear_left = le_v / (le_h + 1e-6) # Evitar división por cero

    # Ojo Derecho (Puntos de MediaPipe: 386 arriba, 374 abajo, 362 izq, 263 der)
    re_v = get_distance(landmarks[386], landmarks[374])
    re_h = get_distance(landmarks[362], landmarks[263])
    ear_right = re_v / (re_h + 1e-6)

    return (ear_left + ear_right) / 2.0


def get_facial_embedding(frame_rgb: np.ndarray, face_detector, face_recognizer, device: torch.device) -> list[float] | None:
    """
    Detects a face, preprocesses it, and extracts a 512D embedding vector.
    Returns None if no face is detected.
    """
    
    # 1. Detect face using MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = face_detector.detect(mp_image)
    
    if not detection_result.detections:
        return None
        
    # 2. Extract bounding box of the most prominent face
    detection = detection_result.detections[0]
    bbox = detection.bounding_box
    
    h, w, _ = frame_rgb.shape
    x_min = max(0, int(bbox.origin_x))
    y_min = max(0, int(bbox.origin_y))
    x_max = min(w, int(bbox.origin_x + bbox.width))
    y_max = min(h, int(bbox.origin_y + bbox.height))
    
    # 3. Crop the face from the original frame
    face_crop = frame_rgb[y_min:y_max, x_min:x_max]
    
    if face_crop.size == 0:
        return None

    # 4. Preprocess for FaceNet (160x160 resolution is mandatory)
    face_resized = cv2.resize(face_crop, (160, 160))
    
    # 5. Convert to PyTorch Tensor and Normalize
    # FaceNet expects inputs normalized between -1 and 1
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Add batch dimension and move to GPU/CPU
    face_tensor = preprocess(face_resized).unsqueeze(0).to(device)
    
    # 6. Extract the 512-dimensional embedding (disable gradients for speed)
    with torch.no_grad():
        embedding_tensor = face_recognizer(face_tensor)
        
    # 7. Convert to standard Python list of floats 
    # (This format is explicitly required by SQLModel and pgvector)
    embedding_list = embedding_tensor[0].cpu().numpy().tolist()
    
    return embedding_list