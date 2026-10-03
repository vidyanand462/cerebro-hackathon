@echo off
title Push ProcrastiGuard AI to GitHub
echo =========================================================
echo       ProcrastiGuard AI - GitHub One-Click Deploy
echo =========================================================
echo.
echo Please enter your GitHub Repository URL (e.g. https://github.com/username/procrastiguard-ai.git):
set /p REPO_URL="Repository URL: "

if "%REPO_URL%"=="" (
    echo [ERROR] No URL entered. Exiting...
    pause
    exit /b
)

echo.
echo [1/4] Checking Git repository...
git init
git add .
git commit -m "feat: initial commit ProcrastiGuard AI for CEREBRO Hackathon"

echo.
echo [2/4] Setting default branch to main...
git branch -M main

echo.
echo [3/4] Linking remote repository...
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo [4/4] Pushing code to GitHub...
git push -u origin main

echo.
echo =========================================================
echo  [SUCCESS] Code pushed to GitHub!
echo  You can now link it on share.streamlit.io for free live deployment!
echo =========================================================
pause
