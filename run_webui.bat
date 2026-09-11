@echo off
title DMOSpeech 2 Web Studio
echo ======================================================
echo           Starting DMOSpeech 2 Web Studio
echo ======================================================
echo.
cd /d "%~dp0"
.\venv\Scripts\python.exe app_gradio.py
pause
