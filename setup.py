import setuptools

setuptools.setup(
    name="editable_gnn",
    version='0.0.0',
    packages=setuptools.find_packages(exclude=('tests',)),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)