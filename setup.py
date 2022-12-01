from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "pandas",
    "torch>=1.8.0",
    "torchvision",
    "tqdm",
    "ipython",
    "graphviz"
]

setup(
    name='torchlens',
    version='0.1.0',
    description="A package for extracting and understanding PyTorch model activations with minimal code",
    long_description=readme,
    author="JohnMark Taylor",
    author_email='johnmarkedwardtaylor@gmail.com',
    url='https://github.com/johnmarktaylor91/torchlens',
    packages=['torchlens'],
    include_package_data=True,
    install_requires=requirements,
    license="GNU GPL v3",
    zip_safe=False,
    keywords='torch torchlens features',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU GPL v3',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9'
    ],
)
