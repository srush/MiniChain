import pathlib

from setuptools import find_packages, setup

LICENSE: str = "MIT"
README: str = pathlib.Path("README.md").read_text(encoding="utf-8")

LIBNAME: str = "minichain"

setup(
    name=LIBNAME,
    version="0.3.1",
    packages=find_packages(
        include=["minichain", "minichain*"],
        exclude=["examples", "docs", "test*"],
    ),
    description="A tiny library for large language models",
    extras_require={},
    long_description=README,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={"minichain": ["templates/*.tpl"]},
    author="Sasha Rush",
    author_email="srush.research@gmail.com",
    url="https://github.com/srush/minichain",
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
    install_requires=[
        "manifest-ml",
        "datasets",
        "gradio",
        "faiss-cpu",
        "eliot",
        "eliot-tree",
        "google-search-results",
        "jinja2",
        "jinja2-highlight",
        "openai==0.28",
        "trio",
    ],
)
