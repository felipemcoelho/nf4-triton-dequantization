from setuptools import setup, find_packages

setup(
    name="nf4-triton-dequantization",
    version="0.1.0",
    description="Optimized Triton kernel for dequantizing NF4 tensors",
    author="Felipe Coelho",
    author_email="felipemc@live.com",
    url="https://github.com/felipemcoelho/nf4-triton-dequantization",
    packages=find_packages(),
    install_requires=[
        "torch",
        "triton",
        "bitsandbytes",
        "unsloth",
        "peft",
        "transformers",
        "numpy",
        "matplotlib",
        "tabulate",
    ],
    extras_require={
        "benchmark": [
            "matplotlib",
            "numpy",
            "tabulate",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    license="Apache License 2.0",
)
