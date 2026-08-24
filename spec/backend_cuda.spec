# -*- mode: python ; coding: utf-8 -*-

import os

_use_upx = os.environ.get("VRCT_PYINSTALLER_UPX") == "1"

_worker_dir = os.path.abspath('./../native/whisper_cpp_worker/stage/whisper_cpp')
_worker_bins = []
if os.path.isdir(_worker_dir):
    _worker_bins = [(os.path.join(_worker_dir, f), 'whisper_cpp')
                    for f in os.listdir(_worker_dir)
                    if f.lower().endswith(('.exe', '.dll'))]


a = Analysis(
    ['..\\src-python\\mainloop.py'],
    pathex=[],
    binaries=_worker_bins,
    datas=[
        ('./../src-python/models/overlay/fonts', 'fonts/'),
        ('./../src-python/models/translation/translation_settings/prompt', 'translation_settings/prompt/'),
        ('./../src-python/models/translation/translation_settings/languages', 'translation_settings/languages/'),
        ('./../.venv_cuda/Lib/site-packages/zeroconf', 'zeroconf/'),
        ('./../.venv_cuda/Lib/site-packages/openvr', 'openvr/'),
        ('./../.venv_cuda/Lib/site-packages/faster_whisper', 'faster_whisper/'),
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
