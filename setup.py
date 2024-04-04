import numpy
from setuptools import Extension, setup

polyext = Extension(
    "pewpew.lib.polyext",
    sources=["src/polyextmodule.c"],
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(ext_modules=[polyext])
