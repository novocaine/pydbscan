from setuptools import setup, Extension
import numpy
import sys

extra_compile_args = [
    # Optimize yet more on gcc compatible compilers
    "-O3", 
    # Hopefully will enable sse if possible
    "-march=native"
]

if sys.platform == "darwin":
    extra_compile_args.extend([
        # For c++14 (we use std::make_unique)
        "-std=c++1y",
        "-stdlib=libc++",
        # This causes warnings about linking with my python.org 2.7.9,
        # but does not seem to cause problems in practice
        "-mmacosx-version-min=10.7"
    ])
else:
    # Assume gcc, sorry no windows
    extra_compile_args.extend([
        "-std=c++0x",
        "-stdlib=libc++"
    ])

ext_modules = [Extension("dbscan", 
    ["pydbscan.cc"],
    extra_compile_args=extra_compile_args)
]

setup(name="dbscan", 
      version="0.1",
      description="Implementation of DBSCAN for Python",
      author="James Salter",
      author_email="iteration@gmail.com",
      license="LGPL",
      install_requires=[
          "numpy",
      ],
      tests_require=[
          "scikit-learn",
          "nose"
      ],
      test_suite = 'nose.collector',
      ext_modules=ext_modules,
      include_dirs=[numpy.get_include()])
