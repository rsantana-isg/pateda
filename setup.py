#!/usr/bin/env python
"""
PATEDA - Python Algorithms for Estimation of Distribution Algorithms

A comprehensive Python library for Estimation of Distribution Algorithms (EDAs)
supporting discrete, continuous, and permutation-based optimization.

Author: Roberto Santana (roberto.santana@ehu.es)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pateda",
    version="0.1.0",
    author="Roberto Santana",
    author_email="roberto.santana@ehu.es",
    description="Python Algorithms for Estimation of Distribution Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rsantana-isg/pateda",
    packages=find_packages(exclude=["tests", "examples", "benchmarks", "scripts", "docs"]),
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
        "Programming Language :: Python :: 3.12",
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
        "neural": [
            "torch>=2.0.0",
        ],
        "copula": [
            "pyvinecopulib>=0.6.0",
        ],
        "all": [
            "numba>=0.55.0",
            "torch>=2.0.0",
            "pyvinecopulib>=0.6.0",
        ],
    },
)
