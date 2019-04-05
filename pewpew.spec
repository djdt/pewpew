# -*- mode: python -*-

import os.path
import sys

exec(open(os.path.join("pewpew", "__init__.py")).read())

block_cipher = None
excludes = []
if sys.platform not in ["win32", "darwin"]:
    excludes = "pewpew.resource"


a = Analysis(
    [os.path.join("pewpew", "__main__.py")],
    pathex=[os.path.abspath(".")],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
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
    name="pewpew" + "_" + __version__,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    icon="icons/pewpew.ico",
)
