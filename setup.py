#!/usr/bin/env python
"""
PATEDA - Python Algorithms for Estimation of Distribution Algorithms

A Python port of MATEDA-3.0 (Matlab toolbox for Estimation of Distribution Algorithms)

Original MATEDA by Roberto Santana (roberto.santana@ehu.es)
Python port by Claude AI Assistant
"""

from setuptools import setup, find_packages

with open("README_PATEDA.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pateda",
    version="0.1.0",
    author="Roberto Santana (original MATEDA), Claude (Python port)",
    author_email="roberto.santana@ehu.es",
    description="Python Algorithms for Estimation of Distribution Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rsantana-isg/pateda",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pgmpy>=0.1.19",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "performance": [
            "numba>=0.55.0",
        ],
    },
)
