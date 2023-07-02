from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    install_requires = [l.strip() for l in f]
    

setup(
    name='engression',
    version='0.0.0.dev0',
    description='Engression',
    url='https://github.com/xwshen51/engression',
    author='Xinwei Shen',
    author_email='xinwei.shen@stat.math.ethz.ch',
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)