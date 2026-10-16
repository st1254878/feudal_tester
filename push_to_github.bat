@echo off
echo ===================================================
echo   FeUdal Networks - One-Click Push to GitHub
echo ===================================================
echo.
set /p REPO_URL="Enter your GitHub Repository URL: "
if "%REPO_URL%"=="" (
    echo Error: Repository URL cannot be empty.
    pause
    exit /b
)

git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main
echo.
echo Pushing to %REPO_URL% ...
git push -u origin main
echo.
echo ===================================================
echo   Upload complete!
echo ===================================================
pause
