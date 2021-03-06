import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [Extension("mantra.util.ranking_cython",
                        ["mantra/util/ranking_cython.pyx"])]

setup(name='mantra',
      version='1.0',
      description='Weakly Supervised Learning',
      url='https://github.com/durandtibo/mantra-python.git',
      author='Thibaut Durand',
      author_email='durand.tibo@gmail.com',
      license='MIT',
      packages=['mantra', 'mantra.util', 'mantra.util.data', 'mantra.util.solver'],
      zip_safe=False,
      ext_modules=cythonize(extensions),
      include_dirs=[numpy.get_include()],
      install_requires=['numpy', 'cython']
      )
