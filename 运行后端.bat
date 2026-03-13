@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
python rag_backend.py
pause
