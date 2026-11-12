import base64
import io
import logging
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

class FaceEngine:
    _instance: Optional["FaceEngine"] = None
    app: Optional[FaceAnalysis] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceEngine, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        logger.info("Initializing InsightFace buffalo_l ArcFace engine...")
        try:
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            logger.info("InsightFace engine successfully prepared.")
        except Exception as e:
            logger.error("Failed to initialize InsightFace engine: %s", e)
            raise e

    @staticmethod
    def decode_base64_image(image_base64: str) -> np.ndarray:
        """Decode base64 string or data URL to OpenCV BGR image."""
        if "," in image_base64:
            # Strip data:image/...;base64, prefix
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from base64 string.")
        return img

    @staticmethod
    def read_image_bytes(image_bytes: bytes) -> np.ndarray:
        """Decode image bytes into OpenCV BGR format."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes.")
        return img

    @staticmethod
    def extract_frame_from_video(video_path: str, frame_fraction: float = 0.5) -> Optional[np.ndarray]:
        """Extract frame from video file (default middle frame as in baseline)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = max(0, int(total_frames * frame_fraction))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    @staticmethod
    def compute_quality_metrics(img: np.ndarray, face_bbox: np.ndarray, det_score: float) -> float:
        """
        Estimate face quality based on detection confidence, sharpness (Laplacian variance),
        brightness uniformity and relative face resolution.
        Returns a normalized score between 0.0 and 1.0.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness via Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 300.0)
        
        # 2. Illumination / Brightness
        mean_brightness = float(np.mean(gray))
        # Ideal range roughly 80 - 180
        if 80 <= mean_brightness <= 180:
            brightness_score = 1.0
        else:
            diff = min(abs(mean_brightness - 80), abs(mean_brightness - 180))
            brightness_score = max(0.2, 1.0 - (diff / 80.0))
            
        # 3. Relative face size
        x1, y1, x2, y2 = face_bbox
        face_area = max(0, (x2 - x1) * (y2 - y1))
        total_area = h * w
        face_ratio = face_area / (total_area + 1e-6)
        size_score = min(1.0, face_ratio / 0.15)  # Ideal if face is at least 15% of frame
        
        # Weighted combination
        quality = (
            0.40 * float(det_score) +
            0.25 * sharpness_score +
            0.20 * brightness_score +
            0.15 * size_score
        )
        return float(np.clip(quality, 0.0, 1.0))

    def extract_face_embedding(self, img: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Detect the primary face, extract aligned ArcFace 512-d embedding, and compute quality.
        
        Returns:
            embedding (np.ndarray): 512-d normalized float32 vector
            quality_score (float): quality estimate in [0, 1]
            metadata (dict): bbox, detection confidence, landmarks
        """
        if self.app is None:
            self._initialize_model()

        faces = self.app.get(img)
        if not faces:
            raise ValueError("No face detected in the provided image.")

        # Pick the largest face (as established in MSU_AVIS_Baseline.ipynb)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        emb = face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        else:
            raise ValueError("Extracted embedding has near-zero magnitude.")

        quality = self.compute_quality_metrics(img, face.bbox, face.det_score)

        metadata = {
            "bbox": [float(x) for x in face.bbox],
            "det_score": float(face.det_score),
            "kps": face.kps.tolist() if hasattr(face, "kps") and face.kps is not None else [],
            "gender": int(face.gender) if hasattr(face, "gender") and face.gender is not None else None,
            "age": int(face.age) if hasattr(face, "age") and face.age is not None else None,
        }

        return emb, quality, metadata

# Global singleton helper
face_engine = FaceEngine()
