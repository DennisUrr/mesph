from distutils.core import setup, Extension
import numpy
module = Extension('trilinear', sources = ['trilinear.c'], include_dirs=[numpy.get_include()])
 
setup (name = 'PackageName',
       version = '0.1',
       description = 'Trilinear interpolation from a spherical mesh Python3',
       author='Pablo Benitez-Llambay',
       author_email='pbllambay@gmail.com',
       ext_modules = [module])

