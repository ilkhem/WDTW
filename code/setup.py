from distutils.core import setup, Extension
import numpy
from Cython.Build import cythonize

# python setup.py build_ext --inplace

setup(
    name = 'dtw_fast',
    ext_modules = cythonize("dtw_fast.pyx"),
    include_dirs=[numpy.get_include()]
)

#
# setup(
#     ext_modules=[
#         Extension("dtw_fast", ["dtw_fast.c"],
#                   include_dirs=[numpy.get_include()]),
#     ],
# )