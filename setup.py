import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cpi-ets",
    version="0.0.2",
    author="Moritz Weber",
    author_email="moritz.weber@kit.edu",
    description="Copy-Paste Imputation for Energy Time Series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="...tbd...",
    packages=['cpiets'],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'pandas', 'prophet'],
    extras_require={
        'dev': [
            'pylint',
        ]
    },
)
