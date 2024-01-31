from distutils.core import setup, Extension
 
module = Extension('trilinear', sources = ['trilinear.c'])
 
setup (name = 'PackageName',
       version = '0.1',
       description = 'Trilinear interpolation from a spherical mesh',
       author='Pablo Benitez-Llambay',
       author_email='pbllambay@gmail.com',
       ext_modules = [module])

