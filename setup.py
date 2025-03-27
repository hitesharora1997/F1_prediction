from setuptools import setup, find_packages

setup(
    name="f1_predictions",
    version="0.1.0",
    description="Formula 1 Race Prediction Framework",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "fastf1>=2.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "pytest>=7.0.0",
        "click>=8.0.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "f1predict=f1_predictions.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)