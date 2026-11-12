import React, { useState, useEffect } from 'react';
import { LayoutDashboard, ShieldCheck, CheckCircle2, XCircle, AlertTriangle, Database, Cpu, Lock, RefreshCw, Users, Activity } from 'lucide-react';

export default function DashboardPage() {
  const [logs, setLogs] = useState([]);
  const [users, setUsers] = useState([]);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [logsRes, usersRes, statusRes] = await Promise.all([
        fetch('/api/v1/auth/logs'),
        fetch('/api/v1/users'),
        fetch('/api/v1/system/status')
      ]);

      if (logsRes.ok) setLogs(await logsRes.json());
      if (usersRes.ok) setUsers(await usersRes.json());
      if (statusRes.ok) setStatus(await statusRes.json());
    } catch (err) {
      console.error("Dashboard data load error", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const enrolledCount = users.filter(u => u.is_face_enrolled).length;
  const acceptedAttempts = logs.filter(l => l.decision === 'ACCEPT').length;
  const acceptRate = logs.length > 0 ? ((acceptedAttempts / logs.length) * 100).toFixed(1) : 0;

  return (
    <div style={{ maxWidth: '1250px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div className="badge badge-info" style={{ marginBottom: '0.5rem' }}>System Monitoring</div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: 800 }}>Biometric Security Dashboard</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Real-time audit trail, biometric template inventory, and system health status.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={fetchData} disabled={loading}>
          <RefreshCw size={16} className={loading ? "pulse-indicator" : ""} /> Refresh Metrics
        </button>
      </div>

      {/* KPI Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem' }}>
        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>Total Subjects</span>
            <Users size={18} color="var(--primary)" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fff' }}>{users.length}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Registered identities
          </div>
        </div>

        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>Enrolled Faces (AES)</span>
            <Lock size={18} color="var(--cyan)" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--cyan)' }}>{enrolledCount}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            512-d ArcFace templates
          </div>
        </div>

        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>Verifications Evaluated</span>
            <Activity size={18} color="var(--emerald)" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fff' }}>{logs.length}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Logged attempts
          </div>
        </div>

        <div className="glass-card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>Acceptance Rate</span>
            <CheckCircle2 size={18} color="var(--emerald)" />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--emerald)' }}>{acceptRate}%</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            {acceptedAttempts} genuine accepts
          </div>
        </div>
      </div>

      {/* System Infrastructure Cards */}
      <div className="glass-card">
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Cpu size={18} color="var(--primary)" /> System Architecture & Security Status
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', fontSize: '0.85rem' }}>
          <div style={{ padding: '0.85rem', backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>Face Recognition Engine:</span>
            <strong style={{ color: '#fff' }}>InsightFace ArcFace</strong>
            <div style={{ marginTop: '0.25rem' }}>
              <span className="badge badge-success">buffalo_l (Active)</span>
            </div>
          </div>

          <div style={{ padding: '0.85rem', backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>Database:</span>
            <strong style={{ color: '#fff' }}>MySQL 8.0 Server</strong>
            <div style={{ marginTop: '0.25rem' }}>
              <span className="badge badge-success">biometric_auth (Connected)</span>
            </div>
          </div>

          <div style={{ padding: '0.85rem', backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>Privacy Engine:</span>
            <strong style={{ color: '#fff' }}>AES-256-GCM AEAD</strong>
            <div style={{ marginTop: '0.25rem' }}>
              <span className="badge badge-info">Zero Plaintext Biometrics</span>
            </div>
          </div>

          <div style={{ padding: '0.85rem', backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(255,255,255,0.05)' }}>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>Face Decision Threshold:</span>
            <strong style={{ color: '#fff' }}>&tau; = 0.4500 (Cosine)</strong>
            <div style={{ marginTop: '0.25rem' }}>
              <span className="badge badge-info">Calibrated on MSU-AVIS</span>
            </div>
          </div>
        </div>
      </div>

      {/* Audit Log Table */}
      <div className="glass-card">
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ShieldCheck size={18} color="var(--cyan)" /> Verification Audit Log ({logs.length})
        </h3>

        {logs.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
            No verification attempts recorded yet.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', textAlign: 'left', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '0.65rem 0.5rem' }}>Timestamp</th>
                  <th style={{ padding: '0.65rem 0.5rem' }}>Attempt Nonce</th>
                  <th style={{ padding: '0.65rem 0.5rem' }}>Subject ID</th>
                  <th style={{ padding: '0.65rem 0.5rem' }}>Cosine Sim</th>
                  <th style={{ padding: '0.65rem 0.5rem' }}>Threshold</th>
                  <th style={{ padding: '0.65rem 0.5rem' }}>Decision</th>
                  <th style={{ padding: '0.65rem 0.5rem' }}>Risk Level</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => (
                  <tr key={log.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <td style={{ padding: '0.65rem 0.5rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                      {new Date(log.created_at).toLocaleTimeString()}
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem' }}>
                      <code style={{ fontSize: '0.72rem', color: 'var(--cyan)' }}>
                        {log.attempt_id.substring(0, 8)}...
                      </code>
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem', fontWeight: 600, color: '#fff' }}>
                      {log.user_id}
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem', fontFamily: 'monospace', fontWeight: 600 }}>
                      {log.similarity_score.toFixed(4)}
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem', color: 'var(--text-dim)', fontFamily: 'monospace' }}>
                      {log.threshold.toFixed(2)}
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem' }}>
                      <span className={`badge ${log.decision === 'ACCEPT' ? 'badge-success' : (log.decision === 'STEP_UP' ? 'badge-warning' : 'badge-danger')}`}>
                        {log.decision}
                      </span>
                    </td>
                    <td style={{ padding: '0.65rem 0.5rem' }}>
                      <span className={`badge ${log.risk_level === 'LOW' ? 'badge-success' : (log.risk_level === 'MEDIUM' ? 'badge-warning' : 'badge-danger')}`}>
                        {log.risk_level}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
