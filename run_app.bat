@echo off
title ProcrastiGuard AI - Academic Procrastination Detector
echo =========================================================
echo       Starting ProcrastiGuard AI Web Dashboard...
echo =========================================================
echo.
python -m streamlit run app/app.py --server.port 8502
pause
