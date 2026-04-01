from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import timm
import torch
import torch.nn as nn


class ModelManager:
    def __init__(self):
        self.face_landmarker = None
        self.face_detector = None
        self.liveness_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self):
        print("⏳ [ModelManager] Loading AI models into memory...")
        loaders = [self.load_face_detector, self.load_face_landmarker, self.load_liveness_model]
        for loader in loaders:
            loader()
        print("✅ [ModelManager] AI models loaded into memory...")
    
    def load_face_detector(self):
        # Initialize MediaPipe Face Detection (Modern Tasks API)
        base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.7)
        self.face_detector = vision.FaceDetector.create_from_options(options)

    def load_face_landmarker(self):
        # Initialize Face Landmarker
        base_options_v2 = python.BaseOptions(model_asset_path='face_landmarker.task')
        options_v2 = vision.FaceLandmarkerOptions(base_options=base_options_v2, num_faces=1)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_v2)

    def load_liveness_model(self):
        # 1. Rebuild the exact same skeleton (without downloading ImageNet weights)
        model = timm.create_model("efficientnet_b0", pretrained=False)
        num_in_features = model.get_classifier().in_features
        model.classifier = nn.Linear(num_in_features, 2)

        # 2. Load the trained weights from disk
        model.load_state_dict(torch.load("liveness_efficientnet.pth", map_location=self.device))

        # 3. Set the model to evaluation mode (disables dropout and batch normalization updates)
        model.eval()
        self.liveness_model = model.to(self.device)
    
    def unload_models(self):
        print("🧹 [ModelManager] Freeing models from memory...")
        self.face_landmarker = None
        self.face_detector = None
        self.liveness_model = None

        # Force PyTorch to release unreferenced GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

ml_manager = ModelManager()