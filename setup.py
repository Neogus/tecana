from setuptools import setup, find_packages
import os

# Load shorter README for PyPI
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README_PYPI.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tecana",
    version="1.1.0",
    author="Gustavo Rabino",
    author_email="gusrab@gmail.com",
    description="A high-performance technical analysis library for financial markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neogus/tecana",
    project_urls={
        "Bug Tracker": "https://github.com/neogus/tecana/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Development Status :: 4 - Beta",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
)
