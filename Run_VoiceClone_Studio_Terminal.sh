#!/bin/bash
# ==============================================================================
# VoxCPM2 Voice Cloning Studio (48kHz) - macOS Launcher
# By Csophouret Dubbing Suite
# ==============================================================================

# Change working directory to this script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "======================================================="
echo " 🎙️  VoxCPM2 Voice Cloning Studio (48kHz) - macOS"
echo " Developed by Csophouret Dubbing Suite"
echo "======================================================="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ រកមិនឃើញ Python 3 ទេ! សូមដំឡើង Python 3 ជាមុនសិន។"
    echo "💡 Please install Python 3 from https://www.python.org/downloads/mac-osx/"
    echo "   or run: brew install python"
    read -p "Press Enter to exit..."
    exit 1
fi

# Create local virtual environment if not exists
if [ ! -d ".venv_mac" ]; then
    echo "⚙️  កំពុងរៀបចំ Virtual Environment សម្រាប់ដំណើរការលើកដំបូង (.venv_mac)..."
    python3 -m venv .venv_mac
fi

if [ -f ".venv_mac/bin/python3" ]; then
    PYTHON_BIN=".venv_mac/bin/python3"
else
    PYTHON_BIN="python3"
fi

# Ensure pip & install dependencies if needed
echo "📦 ត្រួតពិនិត្យបណ្ណាល័យជំនួយ (Checking dependencies)..."
$PYTHON_BIN -m pip install --upgrade pip > /dev/null 2>&1
$PYTHON_BIN -m pip install -r requirements_mac.txt

# Ensure cache and output folders exist
mkdir -p cache_audio
mkdir -p output

echo "🚀 កំពុងដំណើរការកម្មវិធី VoxCPM2 Voice Cloning Studio..."
$PYTHON_BIN voice_clone_main.py

echo "👋 កម្មវិធីត្រូវបានបិទ។"
