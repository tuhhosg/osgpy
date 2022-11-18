#!/usr/bin/python3

from setuptools import setup

reqs = []
with open("requirements.txt") as fd:
    for line in fd.readlines():
        if '#' in line:
            line = line[:line.index('#')]
        if line := line.strip():
            reqs.append(line)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='osgpy',
      version='0.1.2',
      description='Python utilities of the Operating System Group',
      url='http://github.com/tuhhosg/osgpy',
      author='Christian Dietrich',
      author_email='christian.dietrich@tuhh.de',
      long_description=long_description,
      long_description_content_type="text/markdown",
      data_files=[('', ['./requirements.txt'])],
      python_requires=">=3.9",
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10'
      ],
      license='MIT',
      packages=['osg'],
      zip_safe=False,
      install_requires=reqs
)
