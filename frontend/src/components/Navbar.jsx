import React, { useEffect, useState } from 'react';
import { Shield, ShieldCheck, UserPlus, ScanFace, CheckCircle2, LayoutDashboard, Cpu } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  const [systemStatus, setSystemStatus] = useState(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/v1/system/status');
        if (res.ok) {
          const data = await res.json();
          setSystemStatus(data);
        }
      } catch (err) {
        console.error("System status check failed", err);
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: 'register', label: '1. Registration', icon: UserPlus },
    { id: 'enroll', label: '2. Face Enrollment', icon: ScanFace },
    { id: 'verify', label: '3. Face Verification', icon: CheckCircle2 },
    { id: 'voice_enroll', label: '4. Voice Enrollment', icon: Cpu },
    { id: 'voice_verify', label: '5. Voice Verification', icon: ShieldCheck },
    { id: 'dashboard', label: 'Security Dashboard', icon: LayoutDashboard },
  ];

  return (
    <header style={{
      borderBottom: '1px solid var(--border-color)',
      backgroundColor: 'rgba(10, 15, 29, 0.85)',
      backdropFilter: 'blur(16px)',
      position: 'sticky',
      top: 0,
      zIndex: 100,
      padding: '0.75rem 2rem'
    }}>
      <div style={{
        maxWidth: '1350px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1rem'
      }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
          <div style={{
            width: '42px',
            height: '42px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #6366f1 0%, #06b6d4 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 15px rgba(99, 102, 241, 0.4)'
          }}>
            <ShieldCheck size={24} color="#fff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.05rem', fontWeight: 800, letterSpacing: '-0.02em', color: '#fff' }}>
              ATMCG Biometric Auth
            </h1>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Risk-Adaptive &bull; Privacy-Preserving Face–Voice Engine
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', gap: '0.35rem', backgroundColor: 'rgba(255, 255, 255, 0.04)', padding: '0.3rem', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)' }}>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                  padding: '0.55rem 0.95rem',
                  borderRadius: '7px',
                  border: 'none',
                  fontSize: '0.85rem',
                  fontWeight: isActive ? 600 : 500,
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  backgroundColor: isActive ? 'var(--primary)' : 'transparent',
                  color: isActive ? '#fff' : 'var(--text-muted)',
                  boxShadow: isActive ? '0 2px 10px rgba(99, 102, 241, 0.4)' : 'none'
                }}
              >
                <Icon size={16} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* System Health Pill */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.45rem',
            padding: '0.35rem 0.75rem',
            borderRadius: '9999px',
            backgroundColor: systemStatus?.status === 'ONLINE' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(244, 63, 94, 0.12)',
            border: `1px solid ${systemStatus?.status === 'ONLINE' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
            fontSize: '0.75rem',
            fontWeight: 600,
            color: systemStatus?.status === 'ONLINE' ? '#34d399' : '#fb7185'
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: systemStatus?.status === 'ONLINE' ? '#10b981' : '#f43f5e'
            }} className="pulse-indicator" />
            <span>{systemStatus?.status === 'ONLINE' ? 'ArcFace & MySQL Online' : 'Connecting...'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
