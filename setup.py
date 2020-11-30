from setuptools import setup, find_packages

setup(
    name="pewpew",
    description="GUI for visualisation and manipulation of LA-ICP-MS data.",
    url="https://github.com/djdt/pewpew",
    author="T. Lockwood",
    author_email="thomas.lockwood@uts.edu.au",
    version="1.1.0",
    packages=find_packages(include=["pewpew", "pewpew.*"]),
    install_requires=[
        "numpy",
        "matplotlib>=3.0.0",
        "pewlib"
        "PySide2",
    ],
    entry_points={"console_scripts": ["pewpew=pewpew.__main__:main"]},
    tests_require=["pytest", "pytest-qt"],
)
