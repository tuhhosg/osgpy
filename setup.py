#!/usr/bin/python3

from setuptools import setup

setup(name='osgpy',
      version='0.1',
      description='Python utilities of the Operating System Group',
      url='http://github.com/luhsra/sra-cli',
      author='Christian Dietrich',
      author_email='christian.dietrich@tuhh.de',
      license='MIT',
      packages=['osg'],
      zip_safe=False,
      install_requires=['pandas', 'plotnine', 'versuchung'],
)
