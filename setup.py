#!/usr/bin/env python3
"""
Setup script for MNEME - Motor de Memoria Neural Mórfica
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mnemosys",
    version="1.0.0",
    author="Esraderey and Raul Cruz Acosta",
    author_email="msc.framework@gmail.com",
    description="Motor de Memoria Neural Mórfica v2.0 - Sistema avanzado con Safetensors, Locks Granulares y Cache Adaptativo",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica",
    project_urls={
        "Bug Reports": "https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica/issues",
        "Source": "https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica",
        "Documentation": "https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica/wiki",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "cupy>=10.0.0",
        ],
        "security": [
            "cryptography>=3.4.0",
            "pycryptodome>=3.15.0",
        ],
        "optimization": [
            "numba>=0.57.0",
            "numba-cuda>=0.57.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
            "torch>=2.0.0",
            "cupy>=10.0.0",
            "cryptography>=3.4.0",
            "pycryptodome>=3.15.0",
            "numba>=0.57.0",
            "numba-cuda>=0.57.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "memory",
        "neural",
        "compression",
        "pytorch",
        "tensor",
        "machine-learning",
        "ai",
        "optimization",
        "security",
        "deduplication",
        "storage",
        "safetensors",
        "locks",
        "lazy-decompression",
        "adaptive-cache",
        "granular-locks",
    ],
)