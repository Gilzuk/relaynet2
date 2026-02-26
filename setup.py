"""Setup configuration for relaynet package."""

from setuptools import setup, find_packages

setup(
    name="relaynet",
    version="1.0.0",
    description="AI-enhanced relay strategies for two-hop digital communication systems",
    packages=find_packages(exclude=["tests*", "checkpoints*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.22",
        "matplotlib>=3.5",
        "scipy>=1.8",
    ],
    extras_require={
        "cgan": ["torch"],
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Communications",
    ],
)
