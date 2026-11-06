# -*- coding: utf-8 -*-
"""
Standalone VoxCPM2 Voice Cloning Studio App
Powered by Csophouret Dubbing Suite & Hardware-Locked License System.
"""
import os
import sys
from pathlib import Path

# Fix Windows high-DPI scaling
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtCore import Qt

from license_manager import LicenseManager
from activation_dialog import ActivationDialog
from voice_clone_dialog import VoiceCloneDialog


def main():
    import ctypes
    try:
        # Force Windows Taskbar to display custom icon for this app model
        myappid = 'csophouret.voxcpm2.voiceclonestudio.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("VoxCPM2 Voice Cloning Studio")

    # Set Application Icon
    app_dir = Path(__file__).resolve().parent
    icon_path = app_dir / "assets" / "voice_clone_logo.ico"
    if not icon_path.exists():
        icon_path = app_dir / "assets" / "voice_clone_logo.png"
    if icon_path.exists():
        app_icon = QIcon(str(icon_path))
        app.setWindowIcon(app_icon)

    # Set Khmer font fallback
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    # 1. License & Activation Check (Hardware-Locked to Machine ID)
    lic_mgr = LicenseManager()
    lic_status = lic_mgr.check_license()

    if not lic_status.get("valid"):
        dlg = ActivationDialog(license_manager=lic_mgr, status_info=lic_status)
        if icon_path.exists():
            dlg.setWindowIcon(QIcon(str(icon_path)))
        if dlg.exec() != 1:
            # User cancelled or closed the activation screen -> exit cleanly
            sys.exit(0)

    # 2. Launch Voice Clone Studio as Standalone Window
    studio_window = VoiceCloneDialog(parent=None, is_standalone=True)
    studio_window.setWindowTitle("🎙️ VoxCPM2 Voice Cloning Studio (48kHz) - Csophouret")
    if icon_path.exists():
        studio_window.setWindowIcon(QIcon(str(icon_path)))
    studio_window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

