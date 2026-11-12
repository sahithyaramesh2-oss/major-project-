import React, { useState, useEffect } from 'react';
import { CheckCircle2, XCircle, AlertTriangle, ShieldAlert, Camera, Upload, ScanFace, Database, RefreshCw, KeyRound } from 'lucide-react';
import CameraCapture from '../components/CameraCapture';
import SamplePicker from '../components/SamplePicker';

export default function FaceVerifyPage({ preselectedUser }) {
  const [userId, setUserId] = useState(preselectedUser || '');
  const [users, setUsers] = useState([]);
  const [captureMode, setCaptureMode] = useState('camera');
  const [capturedImage, setCapturedImage] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [verifyResult, setVerifyResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/v1/users')
      .then(res => res.json())
      .then(data => {
        setUsers(data);
        if (!userId && data.length > 0) {
          const enrolled = data.find(u => u.is_face_enrolled);
          setUserId(enrolled ? enrolled.user_id : data[0].user_id);
        }
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (preselectedUser) {
      setUserId(preselectedUser);
    }
  }, [preselectedUser]);

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

  const handleSampleSelected = (dataUrl) => {
    setCapturedImage(dataUrl);
  };

  const handleVerify = async () => {
    if (!userId.trim()) {
      setError("Please select or enter the claimed Subject ID.");
      return;
    }

    setLoading(true);
    setError(null);
    setVerifyResult(null);

    try {
      let res;
      if (selectedFile && captureMode === 'upload') {
        const formData = new FormData();
        formData.append('user_id', userId.trim());
        formData.append('file', selectedFile);
        res = await fetch('/api/v1/face/verify', {
          method: 'POST',
          body: formData
        });
      } else if (capturedImage) {
        res = await fetch('/api/v1/face/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId.trim(),
            image_base64: capturedImage
          })
        });
      } else {
        throw new Error("No probe face image provided. Capture a photo, upload a clip, or pick a sample.");
      }

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Face verification failed.");
      }

      setVerifyResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1150px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <div className="badge badge-info" style={{ marginBottom: '0.5rem' }}>Module 1: Biometric Verification</div>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 800 }}>Face Identity Verification</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '750px' }}>
          Live 1:1 biometric matching. Decrypts the registered AES-256-GCM template on the backend,
          extracts the probe ArcFace embedding, and computes cosine similarity with risk-adaptive decisioning.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr minmax(360px, 440px)', gap: '1.75rem', alignItems: 'start' }}>
        {/* Left Column: Probe Input */}
        <div className="glass-card">
          <div className="form-group">
            <label className="form-label">Claimed Subject ID *</label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                type="text"
                className="form-input"
                placeholder="Subject ID being claimed"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
              />
              {users.length > 0 && (
                <select
                  className="form-select"
                  style={{ maxWidth: '190px' }}
                  onChange={(e) => setUserId(e.target.value)}
                  value={userId}
                >
                  <option value="">Select Enrolled</option>
                  {users.map(u => (
                    <option key={u.user_id} value={u.user_id}>
                      {u.user_id} {u.is_face_enrolled ? '✓' : '(Not Enrolled)'}
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
              <Upload size={15} /> Upload Probe File
            </button>
          </div>

          {/* Capture Inputs */}
          {captureMode === 'camera' && (
            <div>
              <CameraCapture
                capturedImage={capturedImage}
                setCapturedImage={setCapturedImage}
              />
              <SamplePicker onSelectSample={handleSampleSelected} mode="verify" />
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
                  id="probe-file-input"
                />
                <label htmlFor="probe-file-input" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                  <Upload size={32} color="var(--primary)" />
                  <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Click to select probe photo or video clip</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Supports JPG, PNG, MP4, AVI</span>
                </label>
              </div>

              {selectedFile && (
                <div style={{ fontSize: '0.85rem', color: 'var(--cyan)' }}>
                  Selected Probe: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024).toFixed(1)} KB)
                </div>
              )}

              {filePreview && (
                <div style={{ maxHeight: '250px', overflow: 'hidden', borderRadius: 'var(--radius-sm)', textAlign: 'center' }}>
                  <img src={filePreview} alt="Preview" style={{ maxHeight: '250px', maxWidth: '100%', objectFit: 'contain' }} />
                </div>
              )}

              <SamplePicker onSelectSample={handleSampleSelected} mode="verify" />
            </div>
          )}

          <div style={{ marginTop: '1.25rem' }}>
            <button
              className="btn btn-primary"
              style={{ width: '100%', padding: '0.75rem' }}
              onClick={handleVerify}
              disabled={loading || (!capturedImage && !selectedFile)}
            >
              <ScanFace size={18} />
              {loading ? 'Evaluating Face Biometrics...' : 'Verify Identity'}
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
              <XCircle size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right Column: Verification Result Dashboard */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {verifyResult ? (
            <div className="glass-card" style={{
              borderColor: verifyResult.decision === 'ACCEPT'
                ? 'rgba(16, 185, 129, 0.5)'
                : verifyResult.decision === 'STEP_UP'
                ? 'rgba(245, 158, 11, 0.5)'
                : 'rgba(244, 63, 94, 0.5)'
            }}>
              {/* Decision Header */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                  {verifyResult.decision === 'ACCEPT' && <CheckCircle2 size={32} color="#34d399" />}
                  {verifyResult.decision === 'STEP_UP' && <AlertTriangle size={32} color="#fbbf24" />}
                  {verifyResult.decision === 'REJECT' && <XCircle size={32} color="#fb7185" />}
                  <div>
                    <h3 style={{ fontSize: '1.3rem', fontWeight: 800 }}>
                      {verifyResult.decision === 'ACCEPT' && 'Identity Verified (ACCEPT)'}
                      {verifyResult.decision === 'STEP_UP' && 'Inconclusive (STEP-UP)'}
                      {verifyResult.decision === 'REJECT' && 'Identity Rejected (REJECT)'}
                    </h3>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      {verifyResult.message}
                    </span>
                  </div>
                </div>
              </div>

              {/* Similarity Score Meter */}
              <div style={{ marginBottom: '1.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Cosine Similarity:</span>
                  <span style={{ fontWeight: 700, fontSize: '1.1rem', color: verifyResult.decision === 'ACCEPT' ? '#34d399' : '#fb7185' }}>
                    {verifyResult.similarity_score.toFixed(4)}
                  </span>
                </div>

                <div className="score-meter">
                  <div
                    className="score-fill"
                    style={{
                      width: `${Math.max(0, Math.min(100, (verifyResult.similarity_score + 0.2) / 1.2 * 100))}%`,
                      backgroundColor: verifyResult.decision === 'ACCEPT' ? '#10b981' : (verifyResult.decision === 'STEP_UP' ? '#f59e0b' : '#f43f5e')
                    }}
                  />
                  <div
                    className="score-threshold-marker"
                    style={{ left: `${((verifyResult.threshold + 0.2) / 1.2) * 100}%` }}
                    title={`Threshold: ${verifyResult.threshold}`}
                  />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                  <span>0.00</span>
                  <span>Threshold: {verifyResult.threshold.toFixed(2)}</span>
                  <span>1.00</span>
                </div>
              </div>

              {/* Detailed Metrics Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.82rem', marginBottom: '1.25rem' }}>
                <div style={{ padding: '0.65rem', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)' }}>
                  <span style={{ color: 'var(--text-muted)', display: 'block' }}>Risk Assessment:</span>
                  <span className={`badge ${verifyResult.risk_level === 'LOW' ? 'badge-success' : (verifyResult.risk_level === 'MEDIUM' ? 'badge-warning' : 'badge-danger')}`} style={{ marginTop: '0.25rem' }}>
                    {verifyResult.risk_level} RISK
                  </span>
                </div>

                <div style={{ padding: '0.65rem', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)' }}>
                  <span style={{ color: 'var(--text-muted)', display: 'block' }}>Probe Quality:</span>
                  <strong style={{ color: '#fff', fontSize: '0.95rem' }}>{(verifyResult.quality_score * 100).toFixed(1)}%</strong>
                </div>

                <div style={{ padding: '0.65rem', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)', gridColumn: 'span 2' }}>
                  <span style={{ color: 'var(--text-muted)', display: 'block' }}>Attempt Nonce / UUID:</span>
                  <code style={{ fontSize: '0.72rem', color: 'var(--cyan)' }}>{verifyResult.attempt_id}</code>
                </div>
              </div>

              {/* Privacy Footer */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                <KeyRound size={14} color="var(--primary)" />
                <span>Verified against AES-256-GCM decrypted gallery embedding</span>
              </div>
            </div>
          ) : (
            <div className="glass-card">
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem' }}>How Face Verification Works</h3>
              <ol style={{ fontSize: '0.82rem', color: 'var(--text-muted)', paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                <li>Select the subject you want to authenticate against.</li>
                <li>Capture your face or select a probe sample clip from the dataset.</li>
                <li>The server securely fetches and decrypts the user's ArcFace template.</li>
                <li>The probe embedding is extracted and compared using normalized cosine similarity:
                  <div style={{ margin: '0.3rem 0', fontFamily: 'monospace', color: 'var(--cyan)' }}>
                    S = (E &middot; T) / (||E|| ||T||)
                  </div>
                </li>
                <li>If S &ge; 0.45 &rarr; <strong>ACCEPT</strong>. Impostors yield S &lt; 0.30 &rarr; <strong>REJECT</strong>.</li>
              </ol>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
