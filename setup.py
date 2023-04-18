import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchhacks",
    version="0.0.1",
    description="Hacks for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Thomas Viehmann, MathInf GmbH",
    url="https://lernappar.at/torchhacks",
    install_requires=["torch"],
    packages=setuptools.find_packages(),
)
