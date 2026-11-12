@echo off
echo ==============================================
echo   PhishGuard AI - FastAPI Backend (Port 8001)
echo ==============================================
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8001 --reload
