from setuptools import setup, find_packages
from pathlib import Path

root = Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (root / "README.rst").read_text(encoding="utf-8")

setup(
    name="pygmol",
    version="1.1.1",
    description="A plasma global model in python",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/hanicinecm/pygmol/",
    author="Martin Hanicinec",
    author_email="hanicinecm@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    keywords="plasma physics chemistry global model",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=["scipy==1.7.3", "pyvalem>=2.5.4", "pandas"],
    extras_require={
        "dev": ["pytest-cov", "black", "ipython"],
    },
    project_urls={
        "Bug Reports": "https://github.com/hanicinecm/pygmol/issues",
        "Documentation": (
            "https://github.com/hanicinecm/pygmol/tree/master/docs/doc_index.rst"
        ),
    },
)
