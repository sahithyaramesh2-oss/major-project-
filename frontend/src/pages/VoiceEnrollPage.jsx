import React, { useState, useEffect, useRef } from 'react';
import {
  Mic, MicOff, Upload, Lock, ShieldCheck, CheckCircle2,
  AlertCircle, StopCircle, AudioLines, Database, ArrowRight
} from 'lucide-react';

// ── helpers ────────────────────────────────────────────────────────────────────
function fmtDuration(sec) {
  const m = Math.floor(sec / 60).toString().padStart(2, '0');
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}
function QualityBar({ label, value, unit = '', colorFn }) {
  if (value == null) return null;
  const display = typeof value === 'number' ? value.toFixed(2) : value;
  const color = colorFn ? colorFn(value) : 'var(--cyan)';
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', padding: '0.4rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <strong style={{ color }}>{display}{unit}</strong>
    </div>
  );
}
// ── component ──────────────────────────────────────────────────────────────────
export default function VoiceEnrollPage({ selectedUser, onEnrollSuccess }) {
  const [userId, setUserId] = useState(selectedUser || '');
  const [users, setUsers] = useState([]);
  const [captureMode, setCaptureMode] = useState('mic');   // 'mic' | 'upload'
  const [challengeId, setChallengeId] = useState('');
  const [challengePhrase, setChallengePhrase] = useState('');

  // recording state
  const [recording, setRecording] = useState(false);
  const [recordSec, setRecordSec] = useState(0);
  const [audioBlob, setAudioBlob] = useState(null);

  // upload state
  const [selectedFile, setSelectedFile] = useState(null);

  // request state
  const [loading, setLoading] = useState(false);
  const [enrollResult, setEnrollResult] = useState(null);
  const [error, setError] = useState(null);

  // refs
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const streamRef = useRef(null);

  // ── load users ───────────────────────────────────────────────────────────────
  useEffect(() => {
    fetch('/api/v1/users')
      .then(r => r.json())
      .then(data => {
        setUsers(data);
        if (!userId && data.length > 0) setUserId(data[0].user_id);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (selectedUser) setUserId(selectedUser);
  }, [selectedUser]);

  const fetchEnrollmentChallenge = async () => {
    try {
      const res = await fetch('/api/v1/voice/challenge?purpose=enrollment');
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Could not load an enrollment phrase.');
      setChallengeId(data.challenge_id);
      setChallengePhrase(data.challenge_phrase);
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchEnrollmentChallenge();
  }, []);

  // ── recording ────────────────────────────────────────────────────────────────
  const startRecording = async () => {
    setError(null);
    setAudioBlob(null);
    setEnrollResult(null);
    chunksRef.current = [];
    let stream;
    try {
      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
        throw new Error('Audio recording is not supported by this browser. Please upload an audio file instead.');
      }
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
      const mimeType = typeof MediaRecorder.isTypeSupported === 'function'
        ? mimeTypes.find(type => MediaRecorder.isTypeSupported(type))
        : undefined;
      const mr = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = () => {
        const recordedType = mr.mimeType || chunksRef.current.find(chunk => chunk.type)?.type || 'audio/webm';
        const blob = new Blob(chunksRef.current, { type: recordedType });
        if (blob.size > 0) {
          setAudioBlob(blob);
        } else {
          setError('The recording was empty. Check microphone access and record again.');
        }
        stream.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      };
      mr.start();
      setRecording(true);
      setRecordSec(0);
      timerRef.current = setInterval(() => setRecordSec(s => s + 1), 1000);
    } catch (err) {
      stream?.getTracks().forEach(t => t.stop());
      streamRef.current = null;
      setError(err.message || 'Could not start recording. Check microphone permissions and try again.');
    }
  };

  const stopRecording = () => {
    clearInterval(timerRef.current);
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  // ── file upload ──────────────────────────────────────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setAudioBlob(null);
      setEnrollResult(null);
      setError(null);
    }
  };

  // ── submit ───────────────────────────────────────────────────────────────────
  const handleEnroll = async () => {
    if (!userId.trim()) { setError('Please select or enter a Subject ID.'); return; }
    if (!challengeId) { setError('Generate an enrollment phrase before recording.'); return; }

    const fileToSend = captureMode === 'upload' ? selectedFile : audioBlob;
    if (!fileToSend) {
      setError(captureMode === 'upload'
        ? 'No audio file selected. Please choose a WAV/MP3/OGG/WebM file.'
        : 'No recording found. Record at least 2 seconds of voice first.');
      return;
    }

    setLoading(true);
    setError(null);
    setEnrollResult(null);

    try {
      const formData = new FormData();
      formData.append('user_id', userId.trim());
      formData.append('challenge_id', challengeId);
      const recordedExtension = audioBlob?.type.includes('wav') ? 'wav'
        : audioBlob?.type.includes('mp4') ? 'm4a'
        : audioBlob?.type.includes('ogg') ? 'ogg' : 'webm';
      formData.append('file', fileToSend, captureMode === 'mic' ? `voice.${recordedExtension}` : selectedFile.name);

      const res = await fetch('/api/v1/voice/enroll', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) {
        const detail = typeof data.detail === 'string' ? data.detail : data.detail?.message;
        throw new Error(detail || data.message || 'Voice enrollment failed.');
      }
      setEnrollResult(data);
      setChallengeId('');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ── render ───────────────────────────────────────────────────────────────────
  const hasAudio = !!audioBlob || (captureMode === 'upload' && !!selectedFile);

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Header */}
      <div>
        <div className="badge badge-info" style={{ marginBottom: '0.5rem' }}>Module 2: Voice Enrollment</div>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 800 }}>Voice Identity Enrollment</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '750px' }}>
          Extracts a 192-dimensional voice embedding using SpeechBrain ECAPA-TDNN.
          The template is encrypted with <strong>AES-256-GCM</strong> authenticated encryption before database storage.
          No raw audio is permanently retained.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr minmax(320px, 420px)', gap: '1.75rem', alignItems: 'start' }}>
        {/* Left: Capture Panel */}
        <div className="glass-card">
          {/* Subject selector */}
          <div className="form-group">
            <label className="form-label">Target Subject *</label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                type="text"
                className="form-input"
                placeholder="Enter Subject ID"
                value={userId}
                onChange={e => setUserId(e.target.value)}
              />
              {users.length > 0 && (
                <select
                  className="form-select"
                  style={{ maxWidth: '185px' }}
                  value={userId}
                  onChange={e => setUserId(e.target.value)}
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
          <div className="form-group" style={{ padding: '0.9rem 1rem', borderRadius: 'var(--radius-sm)', backgroundColor: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.25)' }}>
            <label className="form-label">Enrollment phrase — speak this in your recording</label>
            <div style={{ fontWeight: 700, marginBottom: '0.75rem' }}>{challengePhrase || 'Loading phrase…'}</div>
            <button type="button" className="btn btn-secondary" onClick={fetchEnrollmentChallenge}>New enrollment phrase</button>
          </div>

          {/* Mode Switcher */}
          <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '0.75rem' }}>
            <button
              type="button"
              className={`btn ${captureMode === 'mic' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ fontSize: '0.82rem', padding: '0.45rem 0.85rem' }}
              onClick={() => { setCaptureMode('mic'); setSelectedFile(null); setAudioBlob(null); }}
            >
              <Mic size={15} /> Microphone
            </button>
            <button
              type="button"
              className={`btn ${captureMode === 'upload' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ fontSize: '0.82rem', padding: '0.45rem 0.85rem' }}
              onClick={() => { setCaptureMode('upload'); stopRecording(); setAudioBlob(null); }}
            >
              <Upload size={15} /> Upload File
            </button>
          </div>

          {/* Microphone mode */}
          {captureMode === 'mic' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {/* Recording indicator */}
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                gap: '1rem', padding: '2rem', borderRadius: 'var(--radius-md)',
                backgroundColor: recording ? 'rgba(244, 63, 94, 0.08)' : 'rgba(255,255,255,0.02)',
                border: `2px dashed ${recording ? 'rgba(244,63,94,0.4)' : 'rgba(255,255,255,0.12)'}`,
                transition: 'all 0.3s'
              }}>
                {recording ? (
                  <>
                    <div style={{ position: 'relative' }}>
                      <Mic size={40} color="#fb7185" />
                      <span style={{
                        position: 'absolute', top: -4, right: -4, width: 12, height: 12,
                        borderRadius: '50%', backgroundColor: '#f43f5e',
                        animation: 'pulse 1s infinite'
                      }} className="pulse-indicator" />
                    </div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fb7185', fontFamily: 'monospace' }}>
                      {fmtDuration(recordSec)}
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Recording… speak clearly</div>
                  </>
                ) : audioBlob ? (
                  <>
                    <AudioLines size={36} color="var(--emerald)" />
                    <div style={{ fontSize: '0.88rem', color: '#34d399' }}>
                      Recording captured — {fmtDuration(recordSec)}
                    </div>
                  </>
                ) : (
                  <>
                    <MicOff size={36} color="var(--text-dim)" />
                    <div style={{ fontSize: '0.88rem', color: 'var(--text-muted)' }}>
                      Click <strong>Start Recording</strong> and speak for 3–10 seconds
                    </div>
                  </>
                )}
              </div>

              {/* Recording controls */}
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                {!recording ? (
                  <button className="btn btn-primary" style={{ flex: 1 }} onClick={startRecording}>
                    <Mic size={16} /> Start Recording
                  </button>
                ) : (
                  <button className="btn" style={{ flex: 1, backgroundColor: '#f43f5e', color: '#fff' }} onClick={stopRecording}>
                    <StopCircle size={16} /> Stop Recording
                  </button>
                )}
                {audioBlob && !recording && (
                  <button className="btn btn-secondary" onClick={() => { setAudioBlob(null); setRecordSec(0); setEnrollResult(null); }}>
                    Clear
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Upload mode */}
          {captureMode === 'upload' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{
                border: '2px dashed rgba(255,255,255,0.15)', borderRadius: 'var(--radius-md)',
                padding: '2rem', textAlign: 'center', cursor: 'pointer',
                backgroundColor: 'rgba(255,255,255,0.02)'
              }}>
                <input
                  type="file"
                  accept="audio/*,.wav,.mp3,.ogg,.webm,.m4a,.flac"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                  id="voice-upload-input"
                />
                <label htmlFor="voice-upload-input" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                  <Upload size={32} color="var(--primary)" />
                  <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Click to select audio file</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>WAV, MP3, OGG, WebM, FLAC — min 1s, max 30s</span>
                </label>
              </div>

              {selectedFile && (
                <div style={{ fontSize: '0.85rem', color: 'var(--cyan)' }}>
                  Selected: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024).toFixed(1)} KB)
                </div>
              )}

            </div>
          )}

          {/* Enroll button */}
          <div style={{ marginTop: '1.25rem' }}>
            <button
              className="btn btn-primary"
              style={{ width: '100%', padding: '0.75rem' }}
              onClick={handleEnroll}
              disabled={loading || !hasAudio || recording}
            >
              <AudioLines size={18} />
              {loading ? 'Extracting ECAPA-TDNN & Encrypting…' : 'Enroll Voice Identity'}
            </button>
          </div>

          {/* Error */}
          {error && (
            <div style={{
              marginTop: '1rem', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)',
              backgroundColor: 'rgba(244,63,94,0.15)', border: '1px solid rgba(244,63,94,0.3)',
              color: '#fb7185', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem'
            }}>
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right: Result / Info Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {enrollResult ? (
            <div className="glass-card" style={{ borderColor: 'rgba(16,185,129,0.4)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#34d399', marginBottom: '1rem' }}>
                <CheckCircle2 size={24} />
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Voice Enrollment Succeeded</h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.1rem', marginBottom: '1.25rem' }}>
                <QualityBar label="Subject ID" value={enrollResult.user_id} colorFn={() => '#fff'} />
                <QualityBar label="Embedding Model" value="ECAPA-TDNN (SpeechBrain)" colorFn={() => 'var(--cyan)'} />
                <QualityBar label="Model processing" value={enrollResult.model_processing_status || enrollResult.processing_status} colorFn={() => '#34d399'} />
                <QualityBar label="Embedding Dimension" value={`${enrollResult.embedding_dim} floats`} colorFn={() => '#fff'} />
                <QualityBar label="Enrollment phrase" value={enrollResult.phrase_result || 'PASS'} colorFn={() => '#34d399'} />
                <QualityBar label="Quality Score" value={`${(enrollResult.quality_score * 100).toFixed(1)}%`} colorFn={() => '#34d399'} />
                {enrollResult.audio_quality && (
                  <>
                    <QualityBar label="Duration" value={enrollResult.audio_quality.duration_sec} unit=" s" colorFn={v => v >= 2 ? '#34d399' : '#f59e0b'} />
                    <QualityBar label="SNR" value={enrollResult.audio_quality.snr_db} unit=" dB" colorFn={v => v > 10 ? '#34d399' : v > 5 ? '#f59e0b' : '#fb7185'} />
                    <QualityBar label="Clipping Ratio" value={enrollResult.audio_quality.clipping_ratio} colorFn={v => v < 0.01 ? '#34d399' : v < 0.05 ? '#f59e0b' : '#fb7185'} />
                    <QualityBar label="RMS Energy" value={enrollResult.audio_quality.rms_energy} colorFn={v => v > 0.005 ? '#34d399' : '#f59e0b'} />
                  </>
                )}
              </div>

              {/* Encryption badge */}
              <div style={{
                padding: '0.85rem', borderRadius: 'var(--radius-sm)',
                backgroundColor: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.25)',
                display: 'flex', gap: '0.65rem', marginBottom: '1.25rem'
              }}>
                <Lock size={20} color="var(--primary)" style={{ flexShrink: 0 }} />
                <div style={{ fontSize: '0.78rem', color: '#cbd5e1' }}>
                  <strong>Privacy Protected with AES-256-GCM</strong>
                  <p style={{ marginTop: '0.2rem', color: 'var(--text-muted)' }}>
                    Voice template encrypted with 96-bit random nonce and 128-bit auth tag. Raw audio is not permanently stored.
                  </p>
                </div>
              </div>

              {onEnrollSuccess && (
                <button
                  className="btn btn-success"
                  style={{ width: '100%' }}
                  onClick={() => onEnrollSuccess(enrollResult.user_id)}
                >
                  Proceed to Voice Verification <ArrowRight size={16} />
                </button>
              )}
            </div>
          ) : (
            <div className="glass-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <ShieldCheck size={20} color="var(--cyan)" />
                <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Voice Biometric Specification</h3>
              </div>
              <ul style={{ fontSize: '0.82rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.65rem', paddingLeft: '1.2rem' }}>
                <li><strong>ECAPA-TDNN:</strong> Extracts 192-d speaker embedding from raw audio waveform.</li>
                <li><strong>Audio Preprocessing:</strong> Resampled to 16 kHz mono; validates 1–30 s duration.</li>
                <li><strong>Quality Gating:</strong> Checks SNR, clipping ratio, and RMS energy before enrollment.</li>
                <li><strong>AES-256-GCM:</strong> Zero plaintext biometrics in database — same as face Module 1.</li>
                <li><strong>Challenge Phrases:</strong> Whisper (OpenAI tiny) can verify spoken phrases at verification time.</li>
              </ul>
            </div>
          )}

          {/* Supported formats info */}
          <div className="glass-card" style={{ padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <Database size={16} color="var(--text-muted)" />
              <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)' }}>Audio Requirements</span>
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
              <div>Formats: WAV · MP3 · OGG · WebM · FLAC · M4A</div>
              <div>Duration: 1 s minimum · 30 s maximum</div>
              <div>Sample Rate: 16 kHz (auto-resampled)</div>
              <div>Channels: Mono preferred (stereo auto-mixed)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

