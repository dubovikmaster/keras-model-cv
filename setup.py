from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="keras_model_cv",
    version="0.1.1",
    description="Cross-validation for keras models",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license_files="LICENSE",
    dec="README.md",
    author="Pavel Dubovik",
    author_email="geometryk@gmail.com",
    url="https://github.com/dubovikmaster/keras-cv",
    packages=find_packages(),
    include_package_data=True,
    keywords=[
        'keras cross-validate',
        'validation keras models'
        'cross-validation'

    ],
    install_requires=[
        "tensorflow >= 2.0",
        "scikit-learn",
        "pandas",
        "pyyaml",
    ],
    platforms='any'
)
