#!/usr/bin/python3

from setuptools import setup

reqs = []
with open("requirements.txt") as fd:
    for line in fd.readlines():
        if '#' in line:
            line = line[:line.index('#')]
        if line := line.strip():
            reqs.append(line)

# print(reqs)

setup(name='osgpy',
      version='0.1',
      description='Python utilities of the Operating System Group',
      url='http://github.com/tuhhosg/osgpy',
      author='Christian Dietrich',
      author_email='christian.dietrich@tuhh.de',
      license='MIT',
      packages=['osg'],
      zip_safe=False,
      install_requires=reqs
)
