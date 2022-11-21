from setuptools import setup, find_packages


setup(
    name="lakera_clip",
    version="0.1",
    description="Implementation of CLIP without Pytorch",
    author="Daniel Timbrell",
    author_email="dantimbrell@gmail.com",
    license="Open Source",
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    include_package_data=True,
    packages=find_packages(".", exclude=["tests", "notebooks", "docs"]),
)