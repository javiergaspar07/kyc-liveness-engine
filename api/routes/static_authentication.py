from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import cv2
import numpy as np

# Internal module imports
from core.db import get_session
from core.vision_math import get_facial_embedding
from models import UserBiometrics
from ml.models_manager import ml_manager


router = APIRouter(prefix="/api/v1/kyc", tags=["Static Biometric Authentication"])

MATCH_THRESHOLD = 0.4 

@router.post("/login")
async def authenticate_user(
    external_user_id: str = Form(..., description="The ID of the user attempting to authenticate"),
    file: UploadFile = File(..., description="The live frame of the user's face"),
    session: AsyncSession = Depends(get_session)
):
    """
    Verifies a 1:1 identity claim by comparing the live face exclusively 
    against the enrolled embedding of the provided user ID.
    """
    # 1. Fetch the claimed user from the database FIRST
    # This is an O(1) lookup using the standard B-Tree index
    stmt = select(UserBiometrics).where(
        UserBiometrics.external_user_id == external_user_id,
        UserBiometrics.is_active == True
    )
    result = await session.execute(stmt)
    user_record = result.scalar_one_or_none()

    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or biometric access is disabled."
        )

    # 2. Validate and decode the live image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode the image.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 3. Extract the embedding from the live frame
    live_embedding = get_facial_embedding(
        frame_rgb=frame_rgb,
        face_detector=ml_manager.face_detector,
        face_recognizer=ml_manager.face_recognizer,
        device=ml_manager.device
    )

    if live_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the live frame."
        )

    # 4. Perform the mathematical 1:1 comparison locally
    # Convert lists back to NumPy arrays for fast CPU calculation
    db_vector = np.array(user_record.embedding)
    live_vector = np.array(live_embedding)

    # Calculate Cosine Distance manually (1 - Cosine Similarity)
    # distance = 1 - (A dot B / (||A|| * ||B||))
    dot_product = np.dot(db_vector, live_vector)
    norm_db = np.linalg.norm(db_vector)
    norm_live = np.linalg.norm(live_vector)
    
    similarity = dot_product / (norm_db * norm_live)
    distance = 1.0 - similarity
    
    confidence_percentage = round(max(0.0, similarity) * 100, 2)

    # 5. Evaluate Threshold
    if distance > MATCH_THRESHOLD:
        return {
            "status": "rejected",
            "message": "Identity verification failed. Face does not match the enrolled profile.",
            "confidence": f"{confidence_percentage}%"
        }

    return {
        "status": "matched",
        "external_user_id": user_record.external_user_id,
        "confidence": f"{confidence_percentage}%"
    }


@router.post("/signin")
async def enroll_user(
    external_user_id: str = Form(..., description="The unique ID from the main application"),
    file: UploadFile = File(..., description="The user's selfie image"),
    session: AsyncSession = Depends(get_session)
):
    """
    Registers a new user's face into the biometric database.
    """
    # 1. Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload an image."
        )

    # 2. Read and decode the image from bytes
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode the image. File might be corrupted."
        )

    # 3. Convert BGR to RGB (MediaPipe requirement)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 4. Extract the mathematical embedding using our core logic
    embedding = get_facial_embedding(
        frame_rgb=frame_rgb,
        face_detector=ml_manager.face_detector,
        face_recognizer=ml_manager.face_recognizer,
        device=ml_manager.device
    )

    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the provided image. Please ensure clear lighting and visibility."
        )

    # 5. Save to the PostgreSQL database
    new_biometric_record = UserBiometrics(
        external_user_id=external_user_id,
        embedding=embedding,
        model_version="facenet_inception_resnet_v1"
    )
    
    session.add(new_biometric_record)
    
    try:
        await session.commit()
    except IntegrityError:
        # Rollback the transaction if the user_id already exists in the DB
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with ID {external_user_id} is already enrolled in the biometric system."
        )

    return {
        "status": "success",
        "message": "User biometrics successfully registered.",
        "external_user_id": external_user_id,
        "vector_dimensions": len(embedding)
    }