import numpy
from setuptools import Extension, find_packages, setup

polyext = Extension(
    "pewpew.lib.polyext",
    sources=["src/polyextmodule.c"],
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    packages=find_packages(include=["pewpew", "pewpew.*"]),
    ext_modules=[polyext],
)
