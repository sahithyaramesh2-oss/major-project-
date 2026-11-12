import React, { useRef, useState, useEffect } from 'react';
import { Camera, RefreshCw, Check, AlertCircle, VideoOff } from 'lucide-react';

export default function CameraCapture({ onCapture, capturedImage, setCapturedImage }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [streamActive, setStreamActive] = useState(false);
  const [cameraError, setCameraError] = useState(null);

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setStreamActive(true);
      }
    } catch (err) {
      console.warn("Webcam access error:", err);
      setCameraError("Webcam not accessible or permission denied. You can also upload a photo/video file or use sample clips.");
      setStreamActive(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(t => t.stop());
      videoRef.current.srcObject = null;
      setStreamActive(false);
    }
  };

  useEffect(() => {
    if (!capturedImage) {
      startCamera();
    }
    return () => {
      stopCamera();
    };
  }, [capturedImage]);

  const handleSnap = () => {
    if (!videoRef.current) return;
    const canvas = canvasRef.current || document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    setCapturedImage(dataUrl);
    stopCamera();
    if (onCapture) onCapture(dataUrl);
  };

  const handleRetake = () => {
    setCapturedImage(null);
    if (onCapture) onCapture(null);
    startCamera();
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', width: '100%' }}>
      <div className="camera-box">
        {capturedImage ? (
          <img src={capturedImage} alt="Captured face" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }}
            />
            {streamActive && (
              <div className="camera-overlay">
                <div className="camera-guide-oval"></div>
              </div>
            )}
          </>
        )}

        {cameraError && !capturedImage && (
          <div style={{
            position: 'absolute',
            padding: '1.5rem',
            textAlign: 'center',
            backgroundColor: 'rgba(10, 15, 29, 0.9)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <VideoOff size={36} color="var(--amber)" />
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>{cameraError}</p>
            <button className="btn btn-secondary" onClick={startCamera} style={{ marginTop: '0.5rem' }}>
              Retry Camera
            </button>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Camera Controls */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '0.75rem' }}>
        {capturedImage ? (
          <button type="button" className="btn btn-secondary" onClick={handleRetake}>
            <RefreshCw size={16} /> Retake Photo
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleSnap}
            disabled={!streamActive}
          >
            <Camera size={16} /> Capture Face Snapshot
          </button>
        )}
      </div>
    </div>
  );
}
