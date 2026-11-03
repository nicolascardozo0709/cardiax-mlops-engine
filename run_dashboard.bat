@echo off
echo ======================================================================
echo CardiaX -- Launching FastAPI Telemetry Dashboard
echo ======================================================================
echo Open http://127.0.0.1:8000 in your browser...
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
