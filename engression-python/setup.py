import re
from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = [l.strip() for l in f]

with open('engression/__init__.py') as f:
    version = re.search(r'^__version__ = "(.+)"$', f.read(), re.M).group(1)
    

setup(
    name='engression',
    version=version,
    description='Engression Modelling',
    url='https://github.com/xwshen51/engression',
    author='Xinwei Shen and Nicolai Meinshausen',
    author_email='xwshen@uw.edu',
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="BSD 3-Clause License",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)