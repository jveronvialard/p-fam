from setuptools import setup, find_packages

VERSION = "0.0.1"

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="pfam",
    version=VERSION,
    author="Julien Veron Vialard",
    author_email="julien.veronvialard@gmail.com",
    description="Protein classifier",
    packages=find_packages(),
    install_requires=requirements,
)
