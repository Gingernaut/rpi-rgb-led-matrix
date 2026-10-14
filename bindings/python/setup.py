#!/usr/bin/env python

from glob import glob

from setuptools import setup, Extension

# setuptools only rebuilds an extension when one of its sources or depends is
# newer than the built .so. The library is linked in statically, so list it and
# the headers here, or a library-only change would leave the bindings stale.
LIB_DEPENDS = ['../../lib/librgbmatrix.a'] + glob('../../include/*.h')

core_ext = Extension(
    name                = 'core',
    sources             = ['rgbmatrix/core.cpp'],
    include_dirs        = ['../../include'],
    library_dirs        = ['../../lib'],
    libraries           = ['rgbmatrix'],
    depends             = LIB_DEPENDS,
    extra_compile_args  = ["-O3", "-march=native", "-mtune=native", "-flto=2", "-Wall"],
    language            = 'c++'
)

graphics_ext = Extension(
    name                = 'graphics',
    sources             = ['rgbmatrix/graphics.cpp'],
    include_dirs        = ['../../include'],
    library_dirs        = ['../../lib'],
    libraries           = ['rgbmatrix'],
    depends             = LIB_DEPENDS,
    extra_compile_args  = ["-O3", "-march=native", "-mtune=native", "-flto=2", "-Wall"],
    language            = 'c++'
)

setup(
    name                = 'rgbmatrix',
    version             = '0.0.1',
    author              = 'Christoph Friedrich',
    author_email        = 'christoph.friedrich@vonaffenfels.de',
    classifiers         = ['Development Status :: 3 - Alpha'],
    ext_package         = 'rgbmatrix',
    ext_modules         = [core_ext, graphics_ext],
    packages            = ['rgbmatrix']
)
