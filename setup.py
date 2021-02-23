import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spectral-networks",
    version="0.0.1",
    author="Sophia Sanborn, Christian Shewmake",
    author_email="sophia.sanborn@gmail.com",
    description="Pytorch infrastructure for building and training models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sophiaas/torch-tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)