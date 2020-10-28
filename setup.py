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
        "PySide2",
        "pew@git+https://github.com/djdt/pew#egg=pew-4.0.6",
    ],
    entry_points={"console_scripts": ["pewpew=pewpew.__main__:main"]},
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-qt"],
)
