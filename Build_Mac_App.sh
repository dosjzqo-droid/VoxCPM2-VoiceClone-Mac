#!/bin/bash
# ==============================================================================
# Build Standalone VoxCPM2 Voice Cloning Studio.app on macOS
# ==============================================================================
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "🔨 Building native macOS .app with PyInstaller..."
python3 -m pip install pyinstaller -r requirements_mac.txt
pyinstaller VoiceCloneStudio_mac.spec --clean --noconfirm

echo "======================================================="
echo "✅ Build Complete!"
echo "📁 Your native Mac App is located in: dist/VoxCPM2_VoiceClone_Studio.app"
echo "======================================================="
