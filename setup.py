from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "pandas",
    "pytest",
    "tqdm",
    "ipython",
    "graphviz",
]

setup(
    name="torchlens",
    version="0.1.17",
    description="A package for extracting activations from PyTorch models",
    long_description="A package for extracting activations from PyTorch models. Contains functionality for "
                     "extracting model activations, visualizing a model's computational graph, and "
                     "extracting exhaustive metadata about a model.",
    author="JohnMark Taylor",
    author_email="johnmarkedwardtaylor@gmail.com",
    url="https://github.com/johnmarktaylor91/torchlens",
    packages=["torchlens"],
    include_package_data=True,
    install_requires=requirements,
    license="GNU GPL v3",
    zip_safe=False,
    keywords="torch torchlens features",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
    ],
    extras_require={"dev": ["black[jupyter]", "pytest", "pre-commit"]},
)
