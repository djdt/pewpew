from pathlib import Path
from setuptools import setup, find_packages, Extension
import numpy

with open("README.md") as fp:
    long_description = fp.read()

with Path("pewpew", "__init__.py").open() as fp:
    for line in fp:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')


polyext = Extension(
    "pewpew.lib.polyext",
    sources=["src/polyextmodule.c"],
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="pewpew",
    version=version,
    description="GUI for visualisation and manipulation of LA-ICP-MS data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="T. Lockwood",
    author_email="thomas.lockwood@uts.edu.au",
    url="https://github.com/djdt/pewpew",
    project_urls={
        "Documentation": "https://djdt.github.io/pewpew",
        "Source": "https://gtihub.com/djdt/pewpew",
    },
    packages=find_packages(include=["pewpew", "pewpew.*"]),
    install_requires=[
        "numpy!=1.19.4",
        "pewlib>=0.7.1",
        "PySide2",
    ],
    entry_points={"console_scripts": ["pewpew=pewpew.__main__:main"]},
    tests_require=["pytest", "pytest-qt"],
    ext_modules=[polyext],
)
