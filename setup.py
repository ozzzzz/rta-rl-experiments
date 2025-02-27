from setuptools import setup, find_packages

setup(
    name="massing-generator-rl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A reinforcement learning environment for massing generation in construction.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "gym",
        "shapely",
        "torch",  # or 'tensorflow' depending on your choice of framework
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
