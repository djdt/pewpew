# vim: set ft=python:
from pathlib import Path
from importlib.metadata import version

block_cipher = None

a = Analysis(
    [Path("pewpew", "__main__.py")],
    binaries=None,
    datas=[('pewpew/resources')],
    hiddenimports=[],
    hookspath=None,
    runtime_hooks=None,
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
    name="pewpew" + "_" + version("pewpew"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon="app.ico",
)
