@echo off
echo ==============================================
echo   PhishGuard AI - Pytest Test Suite
echo ==============================================
cd backend
python -m pytest tests/test_phishguard.py -v
echo.
pause
