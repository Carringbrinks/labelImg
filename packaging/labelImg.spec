# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for labelImg — Windows exe build.

Run from the project root:
    pyinstaller packaging/labelImg.spec --noconfirm
Output: dist/labelImg/labelImg.exe (+ _internal).
Copy best.onnx next to labelImg.exe (or into dist/labelImg/model/) so the
AI detection feature finds the model on first run.

Note: PyInstaller resolves `pathex` entries against the current working
directory (unlike scripts/datas, which are spec-relative), so all paths
here are made absolute from SPECPATH to be location-independent.
"""
import os
import sys

project_root = os.path.abspath(os.path.join(SPECPATH, '..'))
spec_dir = os.path.abspath(SPECPATH)

# Explicitly ship the VC++ runtime DLLs from the build interpreter's prefix.
# onnxruntime needs a recent (14.40+) runtime while python.org installers
# ship an older one; see packaging/README_Windows打包说明.md.
_VC_DLLS = ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll',
            'msvcp140_1.dll', 'msvcp140_2.dll', 'msvcp140_atomic_wait.dll',
            'msvcp140_codecvt_ids.dll', 'concrt140.dll']
vc_binaries = [(os.path.join(sys.prefix, n), '.')
               for n in _VC_DLLS if os.path.exists(os.path.join(sys.prefix, n))]

a = Analysis(
    [os.path.join(project_root, 'labelImg.py')],
    pathex=[project_root],
    binaries=vc_binaries,
    datas=[(os.path.join(project_root, 'data', 'ai_classes.txt'), 'data')],
    hiddenimports=['onnxruntime.capi._pybind_state'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt6', 'PySide2', 'PySide6', 'tkinter',
              'torch', 'torchvision', 'ultralytics', 'cv2', 'functorch'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='labelImg',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=os.path.join(spec_dir, 'app.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='labelImg',
)
