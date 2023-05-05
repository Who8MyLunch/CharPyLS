import numpy as np
import setuptools

from setuptools import setup
from setuptools.extension import Extension

from Cython.Distutils import build_ext

# Cython extension.
source_files = ['jpeg_ls/_CharLS.pyx',
                'jpeg_ls/CharLS_src/interface.cpp',
                'jpeg_ls/CharLS_src/jpegls.cpp',
                'jpeg_ls/CharLS_src/header.cpp']

include_dirs = ['jpeg_ls/CharLS_src',
                setuptools.distutils.sysconfig.get_python_inc(),
                np.get_include()]

ext = Extension(name='_CharLS',
                sources=source_files,
                language='c++',
                include_dirs=include_dirs)

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext],
)
