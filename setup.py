from setuptools import find_packages, setup

setup(
    name="hello-wordsmith",
    version="0.1.0",
    description="A simple Python package to wrap llama-index RAG CLI over Wordsmith data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://huggingface.co/datasets/derek-at-work/test/",
    author="Derek Johnston",
    author_email="derek@wordsmith.ai",
    license="MIT",
    packages=find_packages(),
    package_data={
        "hello_wordsmith": ["public_wordsmith_dataset/*"],
    },
    install_requires=[
        "chromadb~=0.5.0",
        "llama-index-core==0.10.33",
        "llama-index-llms-openai~=0.1.16",
        "llama-index-embeddings-openai~=0.1.9",
        "llama-index-vector-stores-chroma~=0.1.7",
    ],
    classifiers=[
        "Intended Audience :: End Users",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    entry_points={
        "console_scripts": [
            "hello-wordsmith=hello_wordsmith.wordsmith:main",
        ],
    },
    python_requires=">=3.8",
)
