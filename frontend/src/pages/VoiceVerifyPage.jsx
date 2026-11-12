import React, { useState, useEffect } from 'react';
import {
  Mic, MicOff, Upload, ShieldCheck, CheckCircle2, XCircle,
  AlertTriangle, AudioLines, RefreshCw, KeyRound, FileAudio
} from 'lucide-react';

function fmtDuration(sec) {
  const m = Math.floor(sec / 60).toString().padStart(2, '0');
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

export default function VoiceVerifyPage({ preselectedUser }) {
  const [userId, setUserId] = useState(preselectedUser || '');
  const [users, setUsers] = useState([]);
  const [captureMode, setCaptureMode] = useState('mic');
  const [challengePhrase, setChallengePhrase] = useState('');
  const [challengeId, setChallengeId] = useState('');
  const [audioBlob, setAudioBlob] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [verifyResult, setVerifyResult] = useState(null);
  const [error, setError] = useState(null);
  const [recording, setRecording] = useState(false);
  const [recordSec, setRecordSec] = useState(0);

  const mediaRecorderRef = React.useRef(null);
  const chunksRef = React.useRef([]);
  const timerRef = React.useRef(null);
  const streamRef = React.useRef(null);

  useEffect(() => {
    fetch('/api/v1/users')
      .then(r => r.json())
      .then(data => {
        setUsers(data);
        if (!userId && data.length > 0) {
          const enrolled = data.find(u => u.is_voice_enrolled);
          setUserId(enrolled ? enrolled.user_id : data[0].user_id);
        }
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (preselectedUser) setUserId(preselectedUser);
  }, [preselectedUser]);

  const fetchChallenge = async () => {
    try {
      const res = await fetch('/api/v1/voice/challenge');
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Challenge generation failed.');
      setChallengeId(data.challenge_id);
      setChallengePhrase(data.challenge_phrase);
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchChallenge();
  }, []);

  const stopRecording = () => {
    clearInterval(timerRef.current);
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
  };

  const startRecording = async () => {
    setError(null);
    setAudioBlob(null);
    setVerifyResult(null);
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

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setAudioBlob(null);
      setVerifyResult(null);
      setError(null);
    }
  };

  const handleVerify = async () => {
    if (!userId.trim()) {
      setError('Please select or enter the claimed Subject ID.');
      return;
    }

    const fileToSend = captureMode === 'upload' ? selectedFile : audioBlob;
    if (!fileToSend) {
      setError(captureMode === 'upload'
        ? 'No audio file selected. Please choose a WAV, MP3, OGG, FLAC or WebM clip.'
        : 'No recording found. Record a clear voice sample first.');
      return;
    }

    setLoading(true);
    setError(null);
    setVerifyResult(null);

    try {
      if (!challengeId) {
        setError('Generate a fresh challenge phrase before verification.');
        return;
      }

      const formData = new FormData();
      formData.append('user_id', userId.trim());
      formData.append('challenge_id', challengeId);
      const recordedExtension = audioBlob?.type.includes('wav') ? 'wav'
        : audioBlob?.type.includes('mp4') ? 'm4a'
        : audioBlob?.type.includes('ogg') ? 'ogg' : 'webm';
      formData.append('file', fileToSend, captureMode === 'upload' ? selectedFile.name : `voice.${recordedExtension}`);
      const res = await fetch('/api/v1/voice/verify', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) {
        if (data.processing_status) {
          setVerifyResult(data);
          fetchChallenge();
          return;
        }
        throw new Error(typeof data.detail === 'string' ? data.detail : data.message || 'Voice verification failed.');
      }
      setVerifyResult(data);
      fetchChallenge();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const hasAudio = !!audioBlob || (captureMode === 'upload' && !!selectedFile);

  return (
    <div style={{ maxWidth: '1150px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div>
        <div className="badge badge-info" style={{ marginBottom: '0.5rem' }}>Module 2: Voice Verification</div>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 800 }}>Voice Identity Verification</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '760px' }}>
          Separately checks the speaker against their encrypted ECAPA-TDNN template and verifies the fresh spoken phrase with Whisper.
          These are independent voice signals; replay detection and final face–voice decisions belong to later modules.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr minmax(360px, 440px)', gap: '1.75rem', alignItems: 'start' }}>
        <div className="glass-card">
          <div className="form-group">
            <label className="form-label">Claimed Subject ID *</label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                type="text"
                className="form-input"
                placeholder="Enter subject ID"
                value={userId}
                onChange={e => setUserId(e.target.value)}
              />
              {users.length > 0 && (
                <select
                  className="form-select"
                  style={{ maxWidth: '200px' }}
                  value={userId}
                  onChange={e => setUserId(e.target.value)}
                >
                  <option value="">Select Enrolled</option>
                  {users.map(u => (
                    <option key={u.user_id} value={u.user_id}>
                      {u.user_id} {u.is_voice_enrolled ? '✓' : '(No voice template)'}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </div>

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

          {captureMode === 'mic' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                gap: '1rem', padding: '2rem', borderRadius: 'var(--radius-md)',
                backgroundColor: recording ? 'rgba(244, 63, 94, 0.08)' : 'rgba(255,255,255,0.02)',
                border: `2px dashed ${recording ? 'rgba(244,63,94,0.4)' : 'rgba(255,255,255,0.12)'}`
              }}>
                {recording ? (
                  <>
                    <div style={{ position: 'relative' }}>
                      <Mic size={38} color="#fb7185" />
                      <span style={{ position: 'absolute', top: -4, right: -4, width: 12, height: 12, borderRadius: '50%', backgroundColor: '#f43f5e', animation: 'pulse 1s infinite' }} />
                    </div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fb7185', fontFamily: 'monospace' }}>{fmtDuration(recordSec)}</div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Recording… speak clearly</div>
                  </>
                ) : audioBlob ? (
                  <>
                    <AudioLines size={34} color="var(--emerald)" />
                    <div style={{ fontSize: '0.88rem', color: '#34d399' }}>Voice sample captured — {fmtDuration(recordSec)}</div>
                  </>
                ) : (
                  <>
                    <MicOff size={34} color="var(--text-dim)" />
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Click <strong>Start Recording</strong> and speak for 2–10 seconds</div>
                  </>
                )}
              </div>

              <div style={{ display: 'flex', gap: '0.75rem' }}>
                {!recording ? (
                  <button className="btn btn-primary" style={{ flex: 1 }} onClick={startRecording}><Mic size={16} /> Start Recording</button>
                ) : (
                  <button className="btn" style={{ flex: 1, backgroundColor: '#f43f5e', color: '#fff' }} onClick={stopRecording}><MicOff size={16} /> Stop Recording</button>
                )}
                {audioBlob && !recording && (
                  <button className="btn btn-secondary" onClick={() => { setAudioBlob(null); setRecordSec(0); setVerifyResult(null); }}>Clear</button>
                )}
              </div>
            </div>
          )}

          {captureMode === 'upload' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ border: '2px dashed rgba(255,255,255,0.15)', borderRadius: 'var(--radius-md)', padding: '2rem', textAlign: 'center', backgroundColor: 'rgba(255,255,255,0.02)' }}>
                <input type="file" accept="audio/*,.wav,.mp3,.ogg,.webm,.m4a,.flac" onChange={handleFileChange} style={{ display: 'none' }} id="voice-verify-upload" />
                <label htmlFor="voice-verify-upload" style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                  <Upload size={32} color="var(--primary)" />
                  <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Select a voice sample</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>WAV, MP3, OGG, WebM, FLAC — 1s to 30s</span>
                </label>
              </div>

              {selectedFile && (
                <div style={{ fontSize: '0.85rem', color: 'var(--cyan)' }}>Selected file: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024).toFixed(1)} KB)</div>
              )}
            </div>
          )}

          <div style={{ marginTop: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            <div className="form-group">
              <label className="form-label">Speak this fresh challenge phrase</label>
              <div style={{ padding: '0.8rem', borderRadius: 'var(--radius-sm)', backgroundColor: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.25)', fontWeight: 700 }}>
                {challengePhrase || 'Loading phrase…'}
              </div>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button className="btn btn-secondary" style={{ flex: 1 }} onClick={fetchChallenge}>
                <RefreshCw size={16} /> New challenge
              </button>
              <button className="btn btn-primary" style={{ flex: 2 }} onClick={handleVerify} disabled={loading || !hasAudio || recording || !challengeId}>
                <ShieldCheck size={18} />
                {loading ? 'Checking voice identity…' : 'Verify Voice Identity'}
              </button>
            </div>
          </div>

          {error && (
            <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', backgroundColor: 'rgba(244,63,94,0.15)', border: '1px solid rgba(244,63,94,0.3)', color: '#fb7185', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <XCircle size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {verifyResult ? (
              <div className="glass-card" style={{ borderColor: verifyResult.speaker_identity === 'MATCH' ? 'rgba(16,185,129,0.4)' : 'rgba(244,63,94,0.4)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem', color: verifyResult.speaker_identity === 'MATCH' ? '#34d399' : '#fb7185' }}>
                {verifyResult.speaker_identity === 'MATCH' ? <CheckCircle2 size={26} /> : <XCircle size={26} />}
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Voice processing: {verifyResult.processing_status}</h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', padding: '0.45rem 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Subject ID</span>
                  <strong>{verifyResult.user_id}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', padding: '0.45rem 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Similarity</span>
                  <strong style={{ color: verifyResult.is_match ? '#34d399' : '#fbbf24' }}>{verifyResult.similarity_score == null ? 'Not evaluated' : verifyResult.similarity_score.toFixed(4)}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', padding: '0.45rem 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Threshold</span>
                  <strong>{verifyResult.threshold.toFixed(4)}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', padding: '0.45rem 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Decision</span>
                  <strong>{verifyResult.speaker_identity || 'NOT EVALUATED'}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', padding: '0.45rem 0' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Quality</span>
                  <strong>{verifyResult.audio_quality_status || 'NOT EVALUATED'}{verifyResult.quality_score == null ? '' : ` (${verifyResult.quality_score.toFixed(4)})`}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', padding: '0.45rem 0' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Challenge</span>
                  <strong>{verifyResult.phrase_result || (verifyResult.phrase_matched ? 'PASS' : 'FAIL')}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', padding: '0.45rem 0' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Model processing</span>
                  <strong>{verifyResult.model_processing_status || verifyResult.processing_status}</strong>
                </div>
              </div>

              <div style={{ padding: '0.8rem 0.9rem', borderRadius: 'var(--radius-sm)', backgroundColor: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.25)', marginBottom: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem', marginBottom: '0.45rem' }}>
                  <KeyRound size={15} color="var(--primary)" />
                  <strong style={{ fontSize: '0.8rem' }}>Challenge Phrase Check</strong>
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {verifyResult.challenge_phrase ? `Expected: ${verifyResult.challenge_phrase}` : 'Challenge phrase unavailable.'}
                </div>
                {verifyResult.transcription && (
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                    Transcription: <strong style={{ color: '#e2e8f0' }}>{verifyResult.transcription}</strong>
                  </div>
                )}
                {verifyResult.phrase_similarity != null && (
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
                    Phrase similarity: <strong>{verifyResult.phrase_similarity.toFixed(4)}</strong>
                    {' '}• Match status: <strong style={{ color: verifyResult.phrase_matched ? '#34d399' : '#fbbf24' }}>{verifyResult.phrase_matched ? 'MATCHED' : 'FAILED'}</strong>
                  </div>
                )}
              </div>

              <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                <strong>Result message:</strong> {verifyResult.message}
                {verifyResult.processing_message && <div style={{ marginTop: '0.35rem' }}>{verifyResult.processing_message}</div>}
              </div>
            </div>
          ) : (
            <div className="glass-card">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <ShieldCheck size={20} color="var(--cyan)" />
                <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Verification Evidence</h3>
              </div>
              <ul style={{ fontSize: '0.82rem', color: 'var(--text-muted)', display: 'flex', flexDirection: 'column', gap: '0.65rem', paddingLeft: '1.2rem' }}>
                <li><strong>Similarity:</strong> Cosine similarity between the enrolled and query voice embeddings.</li>
                <li><strong>Challenge Phrase:</strong> Fresh, single-use phrase verified separately with Whisper.</li>
                <li><strong>Quality:</strong> SNR, clipping ratio, and RMS energy are included in the response.</li>
                <li><strong>Privacy:</strong> Enrolled template is decrypted only in the backend using AES-256-GCM.</li>
              </ul>
            </div>
          )}

          <div className="glass-card" style={{ padding: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <FileAudio size={16} color="var(--text-muted)" />
              <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-muted)' }}>Voice Input Requirements</span>
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
              <div>Formats: WAV · MP3 · OGG · WebM · FLAC · M4A</div>
              <div>Duration: 1s minimum · 30s maximum</div>
              <div>Sample Rate: Resampled to 16kHz mono</div>
              <div>Challenge phrase: required, fresh, and single-use</div>
              <div>Replay detection: not performed by ECAPA-TDNN or Whisper</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
