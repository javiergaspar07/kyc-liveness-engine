import cv2
import numpy as np
import math


def calculate_head_pose(landmarks, img_w, img_h):
    # Extract 6 key anatomical points from MediaPipe
    # 1: Nose tip, 152: Chin, 33: Left eye, 263: Right eye, 61: Left mouth, 291: Right mouth
    image_points = np.array([
        (landmarks[1].x * img_w, landmarks[1].y * img_h),
        (landmarks[152].x * img_w, landmarks[152].y * img_h),
        (landmarks[33].x * img_w, landmarks[33].y * img_h),
        (landmarks[263].x * img_w, landmarks[263].y * img_h),
        (landmarks[61].x * img_w, landmarks[61].y * img_h),
        (landmarks[291].x * img_w, landmarks[291].y * img_h)
    ], dtype="double")

    # Generic 3D human face model coordinates
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals simulation
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))

    # Solve Perspective-n-Point to get rotation vector
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # Convert rotation vector to Euler Angles (Pitch, Yaw, Roll)
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    pitch, yaw, roll = angles[0], angles[1], angles[2]
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