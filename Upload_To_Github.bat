@echo off
chcp 65001 > nul
cls
echo =====================================================================
echo    🚀 កំពុង Upload កូដ Voice Clone Mac ទៅកាន់ GitHub...
echo =====================================================================
echo.

set REPO_URL=https://github.com/dosjzqo-droid/VoxCPM2-VoiceClone-Mac.git

if not exist ".git" (
    git init
    git branch -M main
)

git config user.name "dosjzqo-droid"
git config user.email "dosjzqo@gmail.com"
git add .
git commit -m "Upload Voice Clone Studio for Mac" > nul 2>&1
git remote remove origin > nul 2>&1
git remote add origin %REPO_URL%

echo 🚀 កំពុងរុញកូដទៅកាន់ GitHub (Pushing)...
git push -u origin main

echo.
echo =====================================================================
if %ERRORLEVEL% EQU 0 (
    echo ✅ បាន Upload ទៅកាន់ GitHub ដោយជោគជ័យ!
    echo 👉 សូមត្រឡប់ទៅមើលទំព័រ GitHub រួចចុចលើ Tab "Actions" ដើម្បីទាញយក File .app
) else (
    echo ⚠️ ប្រសិនបើមានបញ្ហា សូមប្រាកដថាបងបាន Sign in លើផ្ទាំង Browser ដែលបានលោតឡើង។
)
echo =====================================================================
pause
