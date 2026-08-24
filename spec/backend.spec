# -*- mode: python ; coding: utf-8 -*-

import os

# UPX compression roughly doubles the PyInstaller collect step and adds
# a few seconds to sidecar startup. Default to disabled so ordinary
# rebuilds are fast; release scripts can opt in by setting
# VRCT_PYINSTALLER_UPX=1.
_use_upx = os.environ.get("VRCT_PYINSTALLER_UPX") == "1"

_worker_dir = os.path.abspath(os.path.join(SPECPATH, '..', 'native', 'whisper_cpp_worker', 'stage', 'whisper_cpp'))
_worker_bins = []
if os.path.isdir(_worker_dir):
    _worker_bins = [(os.path.join(_worker_dir, f), 'whisper_cpp')
                    for f in os.listdir(_worker_dir)
                    if f.lower().endswith(('.exe', '.dll'))]
if not any(os.path.basename(src).lower() == 'vrct-whisper-worker.exe' for src, _ in _worker_bins):
    raise FileNotFoundError(f'whisper.cpp worker was not staged: {_worker_dir}')


a = Analysis(
    ['..\\src-python\\mainloop.py'],
    pathex=[],
    binaries=_worker_bins,
    datas=[
        ('./../src-python/models/overlay/fonts', 'fonts/'),
        ('./../src-python/models/translation/translation_settings/prompt', 'translation_settings/prompt/'),
        ('./../src-python/models/translation/translation_settings/languages', 'translation_settings/languages/'),
        ('./../.venv/Lib/site-packages/zeroconf', 'zeroconf/'),
        ('./../.venv/Lib/site-packages/openvr', 'openvr/'),
        ('./../.venv/Lib/site-packages/faster_whisper', 'faster_whisper/'),
        ('./../.venv/Lib/site-packages/hf_xet', 'hf_xet/')
        ],
    hiddenimports=['faster_whisper.vad', 'models.transcription.audio_pipeline'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pandas', 'matplotlib', 'PyQt5'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VRCT-sidecar-x86_64-pc-windows-msvc',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=_use_upx,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=_use_upx,
    upx_exclude=[],
    name='.',
)
