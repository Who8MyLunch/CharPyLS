
import os
from pathlib import Path
import platform
import setuptools
from setuptools import setup
from setuptools.extension import Extension
from typing import List

from Cython.Build import build_ext, cythonize
import numpy


PACKAGE_DIR = Path(__file__).parent / "jpeg_ls"
LIB_DIR = Path(__file__).parent / "lib" / "charls"
JPEGLS_SRC = LIB_DIR / "src"


def get_source_files() -> List[Path]:
    """Return a list of paths to the source files to be compiled."""
    source_files = [PACKAGE_DIR / "_CharLS.pyx"]
    for fname in JPEGLS_SRC.glob("*"):
        if fname.parts[-1].startswith("design"):
            continue

        if fname.suffix == ".cpp":
            source_files.append(fname)

    return [p.relative_to(Path(__file__).parent) for p in source_files]


# Compiler and linker arguments
extra_compile_args = []
extra_link_args = []
if platform.system() == 'Windows':
    extra_compile_args.append("/Zc:__cplusplus")
    extra_compile_args.append("/Zc:throwingNew")
    extra_compile_args.append("/utf-8")
elif platform.system() in ['Darwin']:
    extra_compile_args.append("-std=c++14")


ext = Extension(
    name="_CharLS",
    sources=[os.fspath(p) for p in get_source_files()],
    language="c++",
    include_dirs=[
        os.fspath(LIB_DIR / "include"),
        os.fspath(JPEGLS_SRC),
        numpy.get_include(),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=[ext],
)
