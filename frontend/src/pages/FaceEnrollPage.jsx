import React, { useState, useEffect } from 'react';
import { ScanFace, Lock, ShieldCheck, CheckCircle2, AlertCircle, Upload, Camera, Database, ArrowRight } from 'lucide-react';
import CameraCapture from '../components/CameraCapture';
import SamplePicker from '../components/SamplePicker';

export default function FaceEnrollPage({ selectedUser, onEnrollSuccess }) {
  const [userId, setUserId] = useState(selectedUser || '');
  const [users, setUsers] = useState([]);
  const [captureMode, setCaptureMode] = useState('camera'); // 'camera', 'upload', 'sample'
  const [capturedImage, setCapturedImage] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [enrollResult, setEnrollResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/v1/users')
      .then(res => res.json())
      .then(data => {
        setUsers(data);
        if (!userId && data.length > 0) {
          setUserId(data[0].user_id);
        }
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (selectedUser) {
      setUserId(selectedUser);
    }
  }, [selectedUser]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onloadend = () => setFilePreview(reader.result);
        reader.readAsDataURL(file);
      } else {
        setFilePreview(null);
      }
    }
  };

  const handleSampleSelected = (dataUrl, sampleUserId) => {
    setCapturedImage(dataUrl);
    if (!userId || userId.startsWith('test_subject')) {
      setUserId(sampleUserId);
    }
  };

  const handleEnroll = async () => {
    if (!userId.trim()) {
      setError("Please select or enter a Subject ID.");
      return;
    }

    setLoading(true);
    setError(null);
    setEnrollResult(null);

    try {
      let res;
      if (selectedFile && captureMode === 'upload') {
        const formData = new FormData();
        formData.append('user_id', userId.trim());
        formData.append('file', selectedFile);
        res = await fetch('/api/v1/face/enroll', {
          method: 'POST',
          body: formData
        });
      } else if (capturedImage) {
        res = await fetch('/api/v1/face/enroll', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId.trim(),
            image_base64: capturedImage
          })
        });
      } else {
        throw new Error("No face image or video selected. Please capture or upload a face.");
      }

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Enrollment failed.");
      }

      setEnrollResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <div className="badge badge-info" style={{ marginBottom: '0.5rem' }}>Module 1: Face Enrollment</div>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 800 }}>Biometric Face Enrollment</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '750px' }}>
          Extracts an aligned 512-dimensional ArcFace identity embedding using InsightFace (buffalo_l).
          The template is encrypted with <strong>AES-256-GCM</strong> authenticated encryption with a unique IV before database storage.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr minmax(320px, 420px)', gap: '1.75rem', alignItems: 'start' }}>
        {/* Left Column: Capture & Source */}
        <div className="glass-card">
          {/* User Selection */}
          <div className="form-group">
            <label className="form-label">Target Subject *</label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                type="text"
                className="form-input"
                placeholder="Enter Subject ID"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
              />
              {users.length > 0 && (
                <select
                  className="form-select"
                  style={{ maxWidth: '180px' }}
                  onChange={(e) => setUserId(e.target.value)}
                  value={userId}
                >
                  <option value="">Choose Existing</option>
                  {users.map(u => (
                    <option key={u.user_id} value={u.user_id}>
                      {u.user_id} ({u.full_name})
                    </option>
                  ))}
                </select>
              )}
            </div>
          </div>

          {/* Mode Switcher */}
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '0.75rem' }}>
            <button
              type="button"
              className={`btn ${captureMode === 'camera' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ fontSize: '0.82rem', padding: '0.45rem 0.85rem' }}
              onClick={() => { setCaptureMode('camera'); setSelectedFile(null); }}
            >
              <Camera size={15} /> Live Webcam
            </button>
            <button
              type="button"
              className={`btn ${captureMode === 'upload' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ fontSize: '0.82rem', padding: '0.45rem 0.85rem' }}
              onClick={() => { setCaptureMode('upload'); setCapturedImage(null); }}
            >
              <Upload size={15} /> Upload File
            </button>
          </div>

          {/* Capture Inputs */}
          {captureMode === 'camera' && (
            <div>
              <CameraCapture
                capturedImage={capturedImage}
                setCapturedImage={setCapturedImage}
              />
              <SamplePicker onSelectSample={handleSampleSelected} mode="enroll" />
            </div>
          )}

          {captureMode === 'upload' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{
                border: '2px dashed rgba(255,255,255,0.15)',
                borderRadius: 'var(--radius-md)',
                padding: '2rem',
                textAlign: 'center',
                cursor: 'pointer',
                backgroundColor: 'rgba(255, 255, 255, 0.02)'
              }}>
                <input
                  type="file"
                  accept="image/*,video/mp4,video/avi"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                  id="file-upload-input"
                />
                <label htmlFor="file-upload-input" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                  <Upload size={32} color="var(--primary)" />
                  <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Click to select face photo or video clip</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Supports JPG, PNG, MP4, AVI</span>
                </label>
              </div>

              {selectedFile && (
                <div style={{ fontSize: '0.85rem', color: 'var(--cyan)' }}>
                  Selected: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024).toFixed(1)} KB)
                </div>
              )}

              {filePreview && (
                <div style={{ maxHeight: '250px', overflow: 'hidden', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                  <img src={filePreview} alt="Preview" style={{ maxHeight: '250px', maxWidth: '100%', objectFit: 'contain' }} />
                </div>
              )}

              <SamplePicker onSelectSample={handleSampleSelected} mode="enroll" />
            </div>
          )}

          {/* Action Button */}
          <div style={{ marginTop: '1.25rem' }}>
            <button
              className="btn btn-primary"
              style={{ width: '100%', padding: '0.75rem' }}
              onClick={handleEnroll}
              disabled={loading || (!capturedImage && !selectedFile)}
            >
              <ScanFace size={18} />
              {loading ? 'Extracting ArcFace & Encrypting...' : 'Enroll Face Identity'}
            </button>
          </div>

          {error && (
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem 1rem',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'rgba(244, 63, 94, 0.15)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              color: '#fb7185',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right Column: Enrollment Status & Cryptographic Safeguards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {enrollResult ? (
            <div className="glass-card" style={{ borderColor: 'rgba(16, 185, 129, 0.4)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#34d399', marginBottom: '1rem' }}>
                <CheckCircle2 size={24} />
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Enrollment Succeeded</h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.85rem' }}>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Subject ID:</span>{' '}
                  <strong style={{ color: '#fff' }}>{enrollResult.user_id}</strong>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>ArcFace Model:</span>{' '}
                  <span className="badge badge-info">InsightFace Buffalo_L</span>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Embedding Dimension:</span>{' '}
                  <strong>{enrollResult.embedding_dim} floats (2048 bytes)</strong>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Quality Score:</span>{' '}
                  <strong style={{ color: '#34d399' }}>{(enrollResult.quality_score * 100).toFixed(1)}%</strong>
                </div>
              </div>

              {/* Encryption Certificate Callout */}
              <div style={{
                marginTop: '1.25rem',
                padding: '0.85rem',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: 'rgba(99, 102, 241, 0.12)',
                border: '1px solid rgba(99, 102, 241, 0.25)',
                display: 'flex',
                gap: '0.65rem'
              }}>
                <Lock size={20} color="var(--primary)" style={{ flexShrink: 0 }} />
                <div style={{ fontSize: '0.78rem', color: '#cbd5e1' }}>
                  <strong>Privacy Protected with AES-256-GCM</strong>
                  <p style={{ marginTop: '0.2rem', color: 'var(--text-muted)' }}>
                    Biometric template is encrypted with a 96-bit random nonce and 128-bit authentication tag. Raw face imagery is not permanently stored.
                  </p>
                </div>
              </div>

              <button
                className="btn btn-success"
                style={{ width: '100%', marginTop: '1.25rem' }}
                onClick={() => onEnrollSuccess(enrollResult.user_id)}
              >
                Proceed to Verification <ArrowRight size={16} />
              </button>
            </div>
          ) : (
            <div className="glass-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <ShieldCheck size={20} color="var(--cyan)" />
                <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Biometric Privacy Specification</h3>
              </div>
              <ul style={{ fontSize: '0.82rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.65rem', paddingLeft: '1.2rem' }}>
                <li><strong>ArcFace Deep Embeddings:</strong> Extracts 512-d normalized facial feature vectors.</li>
                <li><strong>Quality Gating:</strong> Rejects blurry, poorly lit or cropped face inputs before enrollment.</li>
                <li><strong>AES-256-GCM:</strong> Complies with GDPR/privacy guidelines; zero plaintext biometric exposure in MySQL.</li>
                <li><strong>One-Shot Verification:</strong> Cosine similarity matching against decrypted template on-the-fly.</li>
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
