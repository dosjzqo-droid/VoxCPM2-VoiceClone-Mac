# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None
HERE = Path.cwd()

datas = [
    (str(HERE / 'assets'), 'assets'),
    (str(HERE / 'config.json'), '.'),
]

a = Analysis(
    ['voice_clone_main.py'],
    pathex=[str(HERE)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.QtMultimedia',
        'PyQt6.QtSvg',
        'soundfile',
        'numpy',
        'requests',
        'edge_tts',
        'imageio_ffmpeg'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy', 'torch', 'torchaudio'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VoxCPM2_VoiceClone_Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(HERE / 'assets' / 'voice_clone_logo.png'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VoxCPM2_VoiceClone_Studio',
)

app = BUNDLE(
    coll,
    name='VoxCPM2_VoiceClone_Studio.app',
    icon=str(HERE / 'assets' / 'voice_clone_logo.png'),
    bundle_identifier='csophouret.voxcpm2.voiceclonestudio',
)
