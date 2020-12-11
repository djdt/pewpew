from setuptools import setup, find_packages

with open("README.md") as fp:
    long_description = fp.read()

setup(
    name="pewpew",
    version="1.1.1",
    description="GUI for visualisation and manipulation of LA-ICP-MS data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="T. Lockwood",
    author_email="thomas.lockwood@uts.edu.au",
    url="https://github.com/djdt/pewpew",
    project_urls={
        # "Documentation": "https://djdt.github.io/pewlib",
        "Source": "https://gtihub.com/djdt/pewpew",
    },
    packages=find_packages(include=["pewpew", "pewpew.*"]),
    install_requires=[
        "numpy",
        "matplotlib>=3.0.0",
        "pewlib>=0.6.3",
        "PySide2",
    ],
    entry_points={"console_scripts": ["pewpew=pewpew.__main__:main"]},
    tests_require=["pytest", "pytest-qt"],
)
