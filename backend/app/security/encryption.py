import os
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from app.config import settings

def get_cipher_key() -> bytes:
    """Retrieve 32-byte AES key from hex configuration."""
    key_hex = settings.AES_SECRET_KEY_HEX
    try:
        key_bytes = bytes.fromhex(key_hex)
        if len(key_bytes) != 32:
            raise ValueError("AES key must be 32 bytes (256 bits)")
        return key_bytes
    except Exception:
        # Fallback to deterministic 32-byte key derived from secret
        import hashlib
        return hashlib.sha256(key_hex.encode()).digest()

def encrypt_embedding(embedding: np.ndarray, associated_data: bytes = b"biometric_template") -> tuple[bytes, str, str]:
    """
    Encrypt a floating-point biometric embedding using AES-256-GCM.
    
    Returns:
        ciphertext (bytes): The AES-256-GCM encrypted payload
        nonce_hex (str): 12-byte initialization vector in hex
        tag_hex (str): 16-byte authentication tag in hex
    """
    key = get_cipher_key()
    aesgcm = AESGCM(key)
    
    # 12-byte random nonce
    nonce = os.urandom(12)
    
    # Ensure float32 representation and serialize to raw bytes
    raw_bytes = embedding.astype(np.float32).tobytes()
    
    # AESGCM.encrypt appends 16-byte tag to ciphertext
    encrypted_payload = aesgcm.encrypt(nonce, raw_bytes, associated_data)
    ciphertext = encrypted_payload[:-16]
    tag = encrypted_payload[-16:]
    
    return ciphertext, nonce.hex(), tag.hex()

def decrypt_embedding(
    ciphertext: bytes,
    nonce_hex: str,
    tag_hex: str,
    associated_data: bytes = b"biometric_template"
) -> np.ndarray:
    """
    Decrypt and verify an AES-256-GCM encrypted biometric embedding.
    
    Returns:
        np.ndarray: The original normalized 512-d float32 embedding
    """
    key = get_cipher_key()
    aesgcm = AESGCM(key)
    
    nonce = bytes.fromhex(nonce_hex)
    tag = bytes.fromhex(tag_hex)
    
    # Recombine ciphertext and authentication tag
    full_payload = ciphertext + tag
    
    decrypted_bytes = aesgcm.decrypt(nonce, full_payload, associated_data)
    embedding = np.frombuffer(decrypted_bytes, dtype=np.float32)
    return embedding
