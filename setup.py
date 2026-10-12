from setuptools import setup, find_packages

setup(
    name="ssha",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch>=1.12", "numpy>=1.21", "pyyaml>=6.0"],
    python_requires=">=3.8",
)
