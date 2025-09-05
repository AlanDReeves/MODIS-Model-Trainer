from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

# call with python setup.py build_ext --inplace
# clean up with python setup.py clean 
# NOTE: Requires libgcc_s_seh-1.dll, libstdc++-6.dll, and libwinpthread-1.dll when compiling for windows

ext_modules = [
    Extension(
        "SpectraManip",
        ["SpectraManip.cpp"],
        include_dirs = [pybind11.get_include()],
        language="c++",
        extra_compile_args=["/std:c++17"],
    ),
]

setup(
    name="SpectraManip",
    ext_modules=ext_modules,
)
