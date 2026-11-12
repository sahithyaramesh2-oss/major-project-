@echo off
echo Starting ATMCG Biometric Authentication System...
start "ATMCG Backend (FastAPI)" cmd /c "run_backend.bat"
start "ATMCG Frontend (React)" cmd /c "run_frontend.bat"
echo Services launched!
echo Backend:  http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)
echo Frontend: http://localhost:3000
