from typing import Tuple, Dict, Any
import numpy as np
from app.config import settings

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D biometric embedding vectors.
    S = (E · T) / (||E|| ||T||)
    """
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    similarity = np.dot(emb1, emb2) / (norm1 * norm2)
    return float(np.clip(similarity, -1.0, 1.0))

def evaluate_face_verification(
    query_emb: np.ndarray,
    template_emb: np.ndarray,
    quality_score: float = 1.0,
    threshold: float = None
) -> Dict[str, Any]:
    """
    Evaluate verification decision according to cosine similarity and risk thresholds.
    """
    if threshold is None:
        threshold = settings.FACE_SIMILARITY_THRESHOLD

    similarity = compute_cosine_similarity(query_emb, template_emb)
    
    is_match = similarity >= threshold
    
    # Risk-adaptive decision boundaries
    # Strong genuine match
    if similarity >= threshold:
        decision = "ACCEPT"
        risk_level = "LOW"
        message = f"Face verified successfully (Similarity: {similarity:.4f} >= {threshold:.4f})"
    elif similarity >= (threshold - 0.10):
        # Inconclusive / borderline: eligible for step-up challenge
        decision = "STEP_UP"
        risk_level = "MEDIUM"
        message = f"Borderline face match (Similarity: {similarity:.4f}). Step-up verification required."
    else:
        # Strong mismatch / impostor
        decision = "REJECT"
        risk_level = "HIGH"
        message = f"Face verification rejected: identity mismatch (Similarity: {similarity:.4f} < {threshold:.4f})"

    return {
        "similarity_score": round(similarity, 4),
        "threshold": round(threshold, 4),
        "is_match": bool(is_match),
        "decision": decision,
        "risk_level": risk_level,
        "quality_score": round(quality_score, 4),
        "message": message
    }

def evaluate_voice_verification(
    query_emb: np.ndarray,
    template_emb: np.ndarray,
    quality_score: float = 1.0,
    threshold: float = None
) -> Dict[str, Any]:
    """
    Evaluate voice verification evidence according to ECAPA-TDNN cosine similarity.
    Per Module 2 scope: outputs structured biometric identity evidence without
    asserting multimodal risk or anti-spoofing decisions (handled in Modules 3 & 6).
    """
    if threshold is None:
        threshold = settings.VOICE_SIMILARITY_THRESHOLD

    similarity = compute_cosine_similarity(query_emb, template_emb)
    is_match = similarity >= threshold

    if is_match:
        decision = "MATCH"
        message = f"Voice identity verified (Similarity: {similarity:.4f} >= {threshold:.4f})"
    else:
        decision = "MISMATCH"
        message = f"Voice identity mismatch (Similarity: {similarity:.4f} < {threshold:.4f})"

    return {
        "similarity_score": round(similarity, 4),
        "threshold": round(threshold, 4),
        "is_match": bool(is_match),
        "decision": decision,
        "quality_score": round(quality_score, 4),
        "message": message
    }
