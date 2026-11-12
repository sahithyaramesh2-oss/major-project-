import React, { useState, useEffect } from 'react';
import { Database, UserCheck, UserX, Film } from 'lucide-react';

export default function SamplePicker({ onSelectSample, mode = "enroll" }) {
  const [samples, setSamples] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch('/api/v1/system/samples')
      .then(res => res.json())
      .then(data => setSamples(data))
      .catch(err => console.error("Could not fetch samples", err));
  }, []);

  const handlePickSample = async (category, subpath, subjectHint) => {
    setLoading(true);
    try {
      const url = `/api/v1/system/sample-frame/${category}/${encodeURIComponent(subpath)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to load sample frame");
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onloadend = () => {
        onSelectSample(reader.result, subjectHint || subpath);
        setLoading(false);
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.error(err);
      alert("Error loading sample: " + err.message);
      setLoading(false);
    }
  };

  return (
    <div style={{
      backgroundColor: 'rgba(255, 255, 255, 0.03)',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      borderRadius: 'var(--radius-sm)',
      padding: '0.85rem',
      marginTop: '0.75rem'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <Database size={16} color="var(--cyan)" />
        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--cyan)' }}>
          1-Click MSU-AVIS Dataset Samples ({mode === 'enroll' ? 'Gallery' : 'Probe'})
        </span>
      </div>

      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.65rem' }}>
        Test directly with real MSU-AVIS video frames without webcam:
      </p>

      {mode === 'enroll' ? (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ fontSize: '0.78rem', padding: '0.35rem 0.65rem' }}
            disabled={loading}
            onClick={() => handlePickSample('gallery', 'G_F 1.mp4', 'test_subject_01')}
          >
            <Film size={14} /> Subject 01 (Gallery)
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ fontSize: '0.78rem', padding: '0.35rem 0.65rem' }}
            disabled={loading}
            onClick={() => handlePickSample('gallery', 'G_F 2.mp4', 'test_subject_02')}
          >
            <Film size={14} /> Subject 02 (Gallery)
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ fontSize: '0.78rem', padding: '0.35rem 0.65rem' }}
            disabled={loading}
            onClick={() => handlePickSample('gallery', 'G_F 3.mp4', 'test_subject_03')}
          >
            <Film size={14} /> Subject 03 (Gallery)
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ fontSize: '0.78rem', padding: '0.35rem 0.65rem', borderColor: 'rgba(16, 185, 129, 0.4)' }}
            disabled={loading}
            onClick={() => handlePickSample('probe', 'S01_V1/clip_01.mp4', 'test_subject_01')}
          >
            <UserCheck size={14} color="#34d399" /> Genuine: Subj 01 Probe
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ fontSize: '0.78rem', padding: '0.35rem 0.65rem', borderColor: 'rgba(244, 63, 94, 0.4)' }}
            disabled={loading}
            onClick={() => handlePickSample('probe', 'S02_V1/clip_01.mp4', 'test_subject_02')}
          >
            <UserX size={14} color="#fb7185" /> Impostor: Subj 02 Probe
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ fontSize: '0.78rem', padding: '0.35rem 0.65rem', borderColor: 'rgba(244, 63, 94, 0.4)' }}
            disabled={loading}
            onClick={() => handlePickSample('probe', 'S03_V1/clip_01.mp4', 'test_subject_03')}
          >
            <UserX size={14} color="#fb7185" /> Impostor: Subj 03 Probe
          </button>
        </div>
      )}
    </div>
  );
}
