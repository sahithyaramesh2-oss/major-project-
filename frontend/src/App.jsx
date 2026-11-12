import React, { useState } from 'react';
import Navbar from './components/Navbar';
import RegisterPage from './pages/RegisterPage';
import FaceEnrollPage from './pages/FaceEnrollPage';
import FaceVerifyPage from './pages/FaceVerifyPage';
import VoiceEnrollPage from './pages/VoiceEnrollPage';
import VoiceVerifyPage from './pages/VoiceVerifyPage';
import DashboardPage from './pages/DashboardPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('register');
  const [selectedUser, setSelectedUser] = useState('');

  const handleUserSelectForEnroll = (userId) => {
    setSelectedUser(userId);
    setActiveTab('enroll');
  };

  const handleEnrollSuccess = (userId) => {
    setSelectedUser(userId);
    setActiveTab('verify');
  };

  const handleVoiceEnrollSuccess = (userId) => {
    setSelectedUser(userId);
    setActiveTab('voice_verify');
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main style={{ flex: 1, padding: '2rem 1.5rem', maxWidth: '1350px', width: '100%', margin: '0 auto' }}>
        {activeTab === 'register' && (
          <RegisterPage onUserSelectForEnroll={handleUserSelectForEnroll} />
        )}
        {activeTab === 'enroll' && (
          <FaceEnrollPage
            selectedUser={selectedUser}
            onEnrollSuccess={handleEnrollSuccess}
          />
        )}
        {activeTab === 'verify' && (
          <FaceVerifyPage preselectedUser={selectedUser} />
        )}
        {activeTab === 'voice_enroll' && (
          <VoiceEnrollPage
            selectedUser={selectedUser}
            onEnrollSuccess={handleVoiceEnrollSuccess}
          />
        )}
        {activeTab === 'voice_verify' && (
          <VoiceVerifyPage preselectedUser={selectedUser} />
        )}
        {activeTab === 'dashboard' && (
          <DashboardPage />
        )}
      </main>

      <footer style={{
        borderTop: '1px solid var(--border-color)',
        padding: '1.25rem 2rem',
        backgroundColor: 'rgba(10, 15, 29, 0.95)',
        fontSize: '0.8rem',
        color: 'var(--text-dim)',
        textAlign: 'center'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <strong>B.Tech Major Project:</strong> Risk-Adaptive, Privacy-Preserving Face–Voice Biometric Authentication (ATMCG)
          </div>
          <div>
            Stack: InsightFace (ArcFace) &bull; PyTorch &bull; FastAPI &bull; React.js &bull; MySQL &bull; AES-256-GCM
          </div>
        </div>
      </footer>
    </div>
  );
}
