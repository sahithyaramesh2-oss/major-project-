import React, { useState, useEffect } from 'react';
import { UserPlus, Users, CheckCircle2, AlertCircle, ArrowRight, ScanFace } from 'lucide-react';

export default function RegisterPage({ onUserSelectForEnroll }) {
  const [userId, setUserId] = useState('');
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [users, setUsers] = useState([]);

  const loadUsers = async () => {
    try {
      const res = await fetch('/api/v1/users');
      if (res.ok) {
        const data = await res.json();
        setUsers(data);
      }
    } catch (err) {
      console.error("Failed to load users", err);
    }
  };

  useEffect(() => {
    loadUsers();
  }, []);

  const handleRegister = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);

    try {
      const res = await fetch('/api/v1/users/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId.trim(),
          full_name: fullName.trim(),
          email: email.trim() || undefined,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Registration failed');
      }

      setMessage({ type: 'success', text: `Subject "${data.user_id}" registered successfully!` });
      setUserId('');
      setFullName('');
      setEmail('');
      loadUsers();
    } catch (err) {
      setMessage({ type: 'error', text: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Intro Header */}
      <div>
        <div className="badge badge-info" style={{ marginBottom: '0.5rem' }}>Module 1: Identity Registration</div>
        <h2 style={{ fontSize: '1.75rem', fontWeight: 800 }}>User Identity Onboarding</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', maxWidth: '700px' }}>
          Register subjects into the system. Enrolled biometric embeddings are mathematically protected
          using AES-256-GCM authenticated encryption before persisting to MySQL.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 400px) 1fr', gap: '1.75rem', alignItems: 'start' }}>
        {/* Registration Form */}
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginBottom: '1.25rem' }}>
            <UserPlus size={20} color="var(--primary)" />
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700 }}>New Subject</h3>
          </div>

          {message && (
            <div style={{
              padding: '0.75rem 1rem',
              borderRadius: 'var(--radius-sm)',
              marginBottom: '1rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              fontSize: '0.85rem',
              backgroundColor: message.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
              color: message.type === 'success' ? '#34d399' : '#fb7185',
              border: `1px solid ${message.type === 'success' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`
            }}>
              {message.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
              <span>{message.text}</span>
            </div>
          )}

          <form onSubmit={handleRegister}>
            <div className="form-group">
              <label className="form-label">Subject ID (Username) *</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. test_subject_01 or sahithya"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Full Name *</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. Sahithya Ramesh"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Email (Optional)</label>
              <input
                type="email"
                className="form-input"
                placeholder="e.g. sahithya@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <button type="submit" className="btn btn-primary" style={{ width: '100%', marginTop: '0.5rem' }} disabled={loading}>
              <UserPlus size={16} />
              {loading ? 'Registering...' : 'Register Subject'}
            </button>
          </form>
        </div>

        {/* Registered Users Table */}
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
              <Users size={20} color="var(--cyan)" />
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Registered Subjects ({users.length})</h3>
            </div>
            <button className="btn btn-secondary" style={{ fontSize: '0.78rem', padding: '0.35rem 0.65rem' }} onClick={loadUsers}>
              Refresh
            </button>
          </div>

          {users.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '2.5rem 1rem', color: 'var(--text-muted)' }}>
              No subjects registered yet. Create one above to begin.
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', textAlign: 'left', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '0.65rem 0.5rem' }}>Subject ID</th>
                    <th style={{ padding: '0.65rem 0.5rem' }}>Full Name</th>
                    <th style={{ padding: '0.65rem 0.5rem' }}>Face Template</th>
                    <th style={{ padding: '0.65rem 0.5rem', textAlign: 'right' }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => (
                    <tr key={u.user_id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, color: '#fff' }}>
                        <code>{u.user_id}</code>
                      </td>
                      <td style={{ padding: '0.75rem 0.5rem', color: 'var(--text-muted)' }}>{u.full_name}</td>
                      <td style={{ padding: '0.75rem 0.5rem' }}>
                        {u.is_face_enrolled ? (
                          <span className="badge badge-success">Enrolled (AES)</span>
                        ) : (
                          <span className="badge badge-warning">Pending</span>
                        )}
                      </td>
                      <td style={{ padding: '0.75rem 0.5rem', textAlign: 'right' }}>
                        <button
                          className="btn btn-secondary"
                          style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}
                          onClick={() => onUserSelectForEnroll(u.user_id)}
                        >
                          <ScanFace size={14} />
                          {u.is_face_enrolled ? 'Re-enroll' : 'Enroll'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
