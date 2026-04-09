from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select
import cv2
import mediapipe as mp
import numpy as np
import base64
import random

# Internal module imports
from core.db import get_session
from models import UserBiometrics
from ml.models_manager import ml_manager
from core.vision_math import calculate_head_pose, get_facial_embedding

router = APIRouter(prefix="/ws/v3/kyc", tags=["Unified Secure Login"])

MATCH_THRESHOLD = 0.4 
REQUIRED_CHALLENGES = 2 

LIVENESS_CHALLENGES = {
    "turn_right": {
        "instruction": "Please turn your head slowly to the RIGHT.",
        "axis": "yaw",
        "threshold": 25,
        "condition": ">"
    },
    "turn_left": {
        "instruction": "Please turn your head slowly to the LEFT.",
        "axis": "yaw",
        "threshold": -25, 
        "condition": "<"
    },
    "look_up": {
        "instruction": "Please tilt your head slightly UP.",
        "axis": "pitch",
        "threshold": 15,  
        "condition": ">"
    },
    "look_down": {
        "instruction": "Please tilt your head slightly DOWN.",
        "axis": "pitch",
        "threshold": -15, 
        "condition": "<"
    }
}

def get_next_challenge(previous_key: str | None = None) -> str:
    available_keys = list(LIVENESS_CHALLENGES.keys())
    if previous_key in available_keys:
        available_keys.remove(previous_key)
    return random.choice(available_keys)

@router.websocket("/login/{external_user_id}")
async def secure_login_stream(
    websocket: WebSocket, 
    external_user_id: str,
    session: AsyncSession = Depends(get_session)
):
    await websocket.accept()
    print(f"🔒 [SecureLogin] Connection opened for user: {external_user_id}")

    stmt = select(UserBiometrics).where(
        UserBiometrics.external_user_id == external_user_id,
        UserBiometrics.is_active == True
    )
    result = await session.execute(stmt)
    user_record = result.scalar_one_or_none()

    if not user_record:
        await websocket.send_json({"status": "error", "message": "User not found."})
        await websocket.close()
        return

    db_vector = np.array(user_record.embedding)

    challenges_completed = 0
    current_challenge_key = get_next_challenge()
    active_challenge = LIVENESS_CHALLENGES[current_challenge_key]
    
    challenge_action_met = False
    liveness_passed = False

    try:
        await websocket.send_json({
            "status": "challenge_issued",
            "instruction": active_challenge["instruction"],
            "step": f"1/{REQUIRED_CHALLENGES}"
        })

        while True:
            data = await websocket.receive_text()
            frame_data = base64.b64decode(data)
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            ih, iw, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------------------------------------------------------
            # PHASE A: LIVENESS VERIFICATION
            # ---------------------------------------------------------
            if not liveness_passed:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, 
                    data=frame_rgb
                )
                results = ml_manager.face_landmarker.detect(mp_image)

                if results.face_landmarks:
                    landmarks = results.face_landmarks[0]
                    
                    pitch, yaw, roll = calculate_head_pose(landmarks, iw, ih)

                    pose_values = {"pitch": pitch, "yaw": yaw, "roll": roll}
                    current_val = pose_values[active_challenge["axis"]]
                    target_val = active_challenge["threshold"]

                    # Step A.1: Check Action
                    if not challenge_action_met:
                        if active_challenge["condition"] == ">" and current_val > target_val:
                            challenge_action_met = True
                        elif active_challenge["condition"] == "<" and current_val < target_val:
                            challenge_action_met = True
                        
                        if challenge_action_met:
                            await websocket.send_json({
                                "status": "info",
                                "instruction": "Great! Now look straight at the camera."
                            })

                    # Step A.2: Check Return to Center
                    if challenge_action_met:
                        if abs(yaw) < 15 and abs(pitch) < 15:
                            challenges_completed += 1
                            
                            if challenges_completed < REQUIRED_CHALLENGES:
                                current_challenge_key = get_next_challenge(current_challenge_key)
                                active_challenge = LIVENESS_CHALLENGES[current_challenge_key]
                                challenge_action_met = False 
                                
                                await websocket.send_json({
                                    "status": "challenge_issued",
                                    "instruction": active_challenge["instruction"],
                                    "step": f"{challenges_completed + 1}/{REQUIRED_CHALLENGES}"
                                })
                            else:
                                liveness_passed = True
                                await websocket.send_json({
                                    "status": "liveness_passed",
                                    "message": "Liveness confirmed. Authenticating identity..."
                                })
                continue 

            # ---------------------------------------------------------
            # PHASE B: BIOMETRIC AUTHENTICATION
            # ---------------------------------------------------------
            if liveness_passed:
                live_embedding = get_facial_embedding(
                    frame_rgb=frame_rgb,
                    face_detector=ml_manager.face_detector,
                    face_recognizer=ml_manager.face_recognizer,
                    device=ml_manager.device
                )

                if live_embedding is None:
                    await websocket.send_json({"status": "error", "message": "Face lost during capture."})
                    break 

                live_vector = np.array(live_embedding)
                dot_product = np.dot(db_vector, live_vector)
                norm_db = np.linalg.norm(db_vector)
                norm_live = np.linalg.norm(live_vector)
                
                similarity = dot_product / (norm_db * norm_live)
                distance = 1.0 - similarity
                confidence_percentage = round(max(0.0, similarity) * 100, 2)

                if distance > MATCH_THRESHOLD:
                    await websocket.send_json({
                        "status": "rejected",
                        "message": "Identity verification failed.",
                        "confidence": f"{confidence_percentage}%"
                    })
                else:
                    await websocket.send_json({
                        "status": "authenticated",
                        "message": "Welcome back.",
                        "external_user_id": external_user_id,
                        "confidence": f"{confidence_percentage}%"
                    })
                break

    except WebSocketDisconnect:
        print(f"⚠️ [SecureLogin] Client {external_user_id} disconnected unexpectedly.")
    finally:
        await websocket.close()


@router.websocket("/signin/{external_user_id}")
async def dynamic_signin_stream(
    websocket: WebSocket, 
    external_user_id: str,
    session: AsyncSession = Depends(get_session)
):
    """
    Secure Biometric Enrollment (Sign-in): Requires the user to pass a 3D liveness 
    challenge before capturing their facial vector and saving it to the database.
    """
    await websocket.accept()
    print(f"📝 [DynamicSignIn] Enrollment connection opened for user: {external_user_id}")

    # 1. Early DB Check: Ensure user does NOT already exist
    stmt = select(UserBiometrics).where(UserBiometrics.external_user_id == external_user_id)
    result = await session.execute(stmt)
    existing_user = result.scalar_one_or_none()

    if existing_user:
        await websocket.send_json({
            "status": "error", 
            "message": f"User {external_user_id} is already enrolled. Please proceed to login."
        })
        await websocket.close()
        return

    # 2. State Machine Initialization
    challenges_completed = 0
    current_challenge_key = get_next_challenge()
    active_challenge = LIVENESS_CHALLENGES[current_challenge_key]
    
    challenge_action_met = False
    liveness_passed = False

    try:
        # Issue the very first challenge
        await websocket.send_json({
            "status": "challenge_issued",
            "instruction": active_challenge["instruction"],
            "step": f"1/{REQUIRED_CHALLENGES}"
        })

        while True:
            data = await websocket.receive_text()
            frame_data = base64.b64decode(data)
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            ih, iw, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------------------------------------------------------
            # PHASE A: LIVENESS VERIFICATION (Exact same as Login)
            # ---------------------------------------------------------
            if not liveness_passed:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = ml_manager.face_landmarker.detect(mp_image)

                if results.face_landmarks:
                    landmarks = results.face_landmarks[0]
                    pitch, yaw, roll = calculate_head_pose(landmarks, iw, ih)

                    pose_values = {"pitch": pitch, "yaw": yaw, "roll": roll}
                    current_val = pose_values[active_challenge["axis"]]
                    target_val = active_challenge["threshold"]

                    if not challenge_action_met:
                        if active_challenge["condition"] == ">" and current_val > target_val:
                            challenge_action_met = True
                        elif active_challenge["condition"] == "<" and current_val < target_val:
                            challenge_action_met = True
                        
                        if challenge_action_met:
                            await websocket.send_json({
                                "status": "info",
                                "instruction": "Great! Now look straight at the camera to capture your profile."
                            })

                    if challenge_action_met:
                        # Center threshold (Relaxed slightly for natural posture)
                        if abs(yaw) < 15 and abs(pitch) < 15:
                            challenges_completed += 1
                            
                            if challenges_completed < REQUIRED_CHALLENGES:
                                current_challenge_key = get_next_challenge(current_challenge_key)
                                active_challenge = LIVENESS_CHALLENGES[current_challenge_key]
                                challenge_action_met = False 
                                
                                await websocket.send_json({
                                    "status": "challenge_issued",
                                    "instruction": active_challenge["instruction"],
                                    "step": f"{challenges_completed + 1}/{REQUIRED_CHALLENGES}"
                                })
                            else:
                                liveness_passed = True
                                await websocket.send_json({
                                    "status": "liveness_passed",
                                    "message": "Liveness confirmed. Generating biometric profile..."
                                })
                continue 

            # ---------------------------------------------------------
            # PHASE B: BIOMETRIC EXTRACTION AND DATABASE ENROLLMENT
            # ---------------------------------------------------------
            if liveness_passed:
                # Extract the 512D embedding from the perfect center frame
                live_embedding = get_facial_embedding(
                    frame_rgb=frame_rgb,
                    face_detector=ml_manager.face_detector,
                    face_recognizer=ml_manager.face_recognizer,
                    device=ml_manager.device
                )

                if live_embedding is None:
                    await websocket.send_json({"status": "error", "message": "Face lost during final capture."})
                    break 

                # Create the new database record
                new_biometric_record = UserBiometrics(
                    external_user_id=external_user_id,
                    embedding=live_embedding,
                    model_version="facenet_inception_resnet_v1"
                )
                
                session.add(new_biometric_record)
                
                try:
                    await session.commit()
                    await websocket.send_json({
                        "status": "enrolled",
                        "message": "Biometric profile successfully registered.",
                        "external_user_id": external_user_id,
                        "vector_dimensions": len(live_embedding)
                    })
                except IntegrityError:
                    # Failsafe in case of a race condition creating the same user
                    await session.rollback()
                    await websocket.send_json({
                        "status": "error",
                        "message": "Database conflict. User might have been enrolled during this session."
                    })
                
                break # Close the atomic transaction

    except WebSocketDisconnect:
        print(f"⚠️ [DynamicSignIn] Client {external_user_id} disconnected unexpectedly.")
    finally:
        await websocket.close()