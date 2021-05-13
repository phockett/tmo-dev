# Minimal setup.py
# From https://github.com/maet3608/minimal-setup-py
# Plus some mods from Hatch setup.py format.

from io import open

from setuptools import find_packages, setup

with open('tmo/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = []

setup(
    name='tmo-dev',
    version=version,
    url='https://github.com/phockett/tmo-dev',
    author='Paul Hockett',
    author_email='',
    description='SLAC TMO data analysis, and related.',
    packages=find_packages(),
    install_requires=REQUIRES,
)
