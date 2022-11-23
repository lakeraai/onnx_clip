from setuptools import setup, find_packages

base_packages = [
    "numpy>=0.18.0",
    "opencv-python-headless>=4.0.1",
    "onnxruntime>=0.4.0",
    "pillow>=8.4.0",
    "ftfy>=6.0.3"
]

setup(
    name="lakera_clip",
    version="0.1",
    description="Implementation of CLIP without Pytorch",
    author="Daniel Timbrell",
    author_email="dt@lakera.ai",
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