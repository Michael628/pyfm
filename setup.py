from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pyfm",
    author="Michael Lynch",
    author_email="michaellynch628@gmail.com",
    version="0.1.0",
    packages=[
        "pyfm",
        "pyfm.a2a",
        "pyfm.nanny",
        "pyfm.nanny.tasks",
        "pyfm.nanny.tasks.hadrons",
        "pyfm.processing",
    ],
    url="https://github.com/Michael628/pyfm",  # Project URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",  # Python version requirement
    description="Nanny, postprocessing, and A2A contraction scripts",
    install_requires=requirements,
)
