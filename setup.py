import numpy
from Cython.Build import cythonize
from setuptools import setup

setup(name='mantra',
      version='1.0',
      description='Weakly Supervised Learning',
      url='https://gitlab.com/durandtibo/wsltibo',
      author='Thibaut Durand',
      author_email='durand.tibo@gmail.com',
      license='MIT',
      packages=['mantra', 'mantra.util', 'mantra.util.data', 'mantra.util.solver'],
      zip_safe=False,
      ext_modules=cythonize(['mantra/util/ranking_cython.pyx']),
      include_dirs=[numpy.get_include()],
      install_requires=['numpy', 'cython']
      )
