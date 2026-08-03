from setuptools import setup, find_packages

setup(
    name="hermes_vla",
    version="1.0.0",
    description="Hermes-VLA: A Dual-Stream Hierarchical VLA Framework for Embodied Intelligence",
    author="molyswu",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "transformers>=4.40.0",
    ],
    extras_require={
        "lerobot": [
            "lerobot>=0.3.0",
            "datasets>=2.14.0",
            "huggingface_hub>=0.20.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
