# Risk-Adaptive, Privacy-Preserving Face–Voice Biometric Authentication

Using Attack-Conditioned Temporal Multimodal Challenge Generation (ATMCG).

## Project Overview
This project develops an adaptive, privacy-preserving face–voice biometric authentication system based on the research proposal:
1. **Module 1**: Face Identity Verification (InsightFace / ArcFace + Cosine Similarity + AES-256-GCM Template Encryption)
2. **Module 2**: Voice Identity Verification (SpeechBrain ECAPA-TDNN)
3. **Module 3**: Multimodal Liveness Detection (MiniFASNet + AASIST)
4. **Module 4**: Cross-Modal Temporal Consistency (Lip Sync & SyncNet-style timing)
5. **Module 5**: Attack-Conditioned Temporal Multimodal Challenge Generation (ATMCG)
6. **Module 6**: Risk-Adaptive Decision Policy & Privacy Engine

---

## Current Status: Module 1 Completed & Verified

## Module 2: Voice Identity Verification

### Voice module capabilities
- SpeechBrain ECAPA-TDNN speaker embedding extraction
- AES-256-GCM encrypted voice template storage in the same `biometric_templates` table used by face enrollment
- Cosine-similarity voice verification against an enrolled template
- Required Whisper-based enrollment and fresh, single-use verification challenge phrase checks
- Audio validation for empty, corrupted, too-short and unsupported inputs
- Separate speaker identity, phrase compliance, audio quality, and model-processing outcomes
- No synthetic fallback embeddings; failed model inference does not produce a biometric score

### Backend endpoints
- `GET /api/v1/voice/challenge?purpose=enrollment` — returns an expiring enrollment phrase and challenge ID
- `GET /api/v1/voice/challenge?purpose=verification` — returns an expiring, single-use verification phrase and challenge ID
- `POST /api/v1/voice/enroll` — accepts audio and a valid enrollment challenge ID; stores only an encrypted ECAPA-TDNN template
- `POST /api/v1/voice/verify` — checks speaker identity and the server-issued phrase independently

### API and model configuration
- `VOICE_SIMILARITY_THRESHOLD` controls the identity acceptance threshold
- `CHALLENGE_MATCH_THRESHOLD` controls word/phrase acceptance for challenge transcription
- `VOICE_MIN_DURATION_SEC` and `VOICE_MAX_DURATION_SEC` validate audio capture bounds
- `WHISPER_MODEL_SIZE` controls the transcription model size

### Module 2 workflow
1. Issue an expiring enrollment/verification phrase and challenge ID.
2. Validate and preprocess input audio to 16 kHz mono; reject silence, unusable quality, or invalid duration.
3. Use Whisper to confirm the required enrollment or verification phrase.
4. Extract and validate a normalized 192-dimensional ECAPA-TDNN embedding; never substitute a synthetic embedding.
5. Encrypt and store enrollment templates with AES-256-GCM.
6. During verification, retrieve and decrypt the template, then return speaker identity separately from phrase compliance.
7. Leave replay detection and final face–voice decisions to their designated modules. The voice module does not claim to detect replay attacks.

Voice similarity thresholds must be calibrated against representative genuine and impostor validation recordings before production use.

### Implemented Capabilities:
- **Backend Architecture**: FastAPI with asynchronous endpoints, Pydantic validation, and SQLAlchemy ORM.
- **Database**: Production MySQL 8.0 server (`biometric_auth` database) with relational schema for users, encrypted templates, and audit logs.
- **Biometric Model**: Pretrained InsightFace ArcFace (`buffalo_l` model zoo) extracting 512-dimensional normalized embeddings and quality metrics.
- **Privacy Engine**: AES-256-GCM AEAD encryption with 96-bit unique nonce and 128-bit authentication tag for biometric templates.
- **Matching & Decision**: Normalized cosine similarity matching with calibrated risk-adaptive thresholds:
  - Genuine match: $S \ge 0.45$ &rarr; `ACCEPT` (Low Risk)
  - Borderline: $0.35 \le S < 0.45$ &rarr; `STEP_UP` (Medium Risk)
  - Impostor: $S < 0.35$ &rarr; `REJECT` (High Risk)
- **Frontend Application**: Modern React.js (Vite) interface:
  - User Registration screen
  - Face Enrollment screen (Webcam capture, video/image upload, 1-click MSU-AVIS sample loader)
  - Face Verification screen (Live match gauge, decision badge, risk evaluation)
  - Security Dashboard (Live MySQL audit logs, system status, infrastructure health)

---

## Test Results on Real MSU-AVIS Dataset

Tested against real video clips from the MSU-AVIS dataset:
- **Gallery Enrollment**: `G_F 1.mp4` &rarr; 512-d ArcFace embedding encrypted and stored in MySQL.
- **Genuine Verification**: Probe `S01_V1/clip_01.mp4` &rarr; **Cosine Similarity: 0.5395** ($\ge 0.45$) &rarr; **`ACCEPT` (LOW RISK)**.
- **Impostor Verification**: Probe `S02_V1/clip_01.mp4` &rarr; **Cosine Similarity: 0.1122** ($< 0.45$) &rarr; **`REJECT` (HIGH RISK)**.
- **Automated Test Suite**: 6/6 tests passing in `backend/tests/test_face_verification.py`.

---

## How to Run

### 1. Start Both Services
Double click `run_all.bat` or run:
```bash
.\run_all.bat
```

### 2. Manual Commands
**Backend (FastAPI):**
```bash
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```
Swagger API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Frontend (React.js):**
```bash
cd frontend
npm run dev
```
Open browser at [http://localhost:3000](http://localhost:3000)
