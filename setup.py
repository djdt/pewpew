from setuptools import setup, find_packages

from pewpew import __version__

setup(
    name="pewpew",
    description="GUI for visualisation and manipulation of LA-ICP-MS data.",
    author="djdt",
    version=__version__,
    packages=find_packages(include=["pewpew", "pewpew.*"]),
    install_requires=[
        "numpy",
        "matplotlib>=3.0.0",
        "pew>=0.2.7 @ git+https//github.com/djdt/pew@master",
        "PySide2>=5.13.0",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-qt"],
)
