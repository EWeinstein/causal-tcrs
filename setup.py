import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CausalImmuneReceptors",
    version="0.0.1",
    author="Eli N. Weinstein",
    author_email="eli.n.weinstein@gmail.com",
    description="Causal inference for patient immune receptor repertoire sequencing data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[],
    extras_require={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        'Topic :: Scientific/Engineering :: Computational Biology',
    ],
    python_requires='>=3.8',
    keywords=('biological-sequences proteins probabilistic-models ' +
              'machine-learning immunology causal-inference'),
)
