# vim: set ft=python:
import os.path


block_cipher = None

a = Analysis(
    [os.path.join("pewpew", "__main__.py")],
    binaries=None,
    datas=None,
    hiddenimports=[],
    hookspath=None,
    runtime_hooks=None,
    excludes=["FixTk", "tcl", "tk", "_tkinter", "tkinter", "Tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    exclude_binaries=False,
    name="pewpew",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="app.ico",
)
