import pathlib

from setuptools import find_packages, setup

LICENSE: str = "MIT"
README: str = pathlib.Path("README.md").read_text(encoding="utf-8")

LIBNAME: str = "minichain"

setup(
    name=LIBNAME,
    version="0.1",
    packages=find_packages(
        include=["minichain", "minichain*"],
        exclude=["examples", "docs", "test*"],
    ),
    description="A declarative drawing API",
    install_requires=[],
    extras_require={},
    long_description=README,
    long_description_content_type="text/markdown",
    author="Dan Oneață",
    author_email="dan.oneata@gmail.com",
    url="https://github.com/chalk-diagrams/chalk",
    project_urls={
        "Documentation": "https://srush.github.io/minichain",
        "Source Code": "https://github.com/srush/minichain",
        "Issue Tracker": "https://github.com/srush/minichain/issues",
    },
    license=LICENSE,
    license_files=("LICENSE",),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        f"License :: OSI Approved :: {LICENSE} License",
        "Topic :: Scientific/Engineering",
    ],
)
