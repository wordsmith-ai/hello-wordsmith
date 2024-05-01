from setuptools import setup, find_packages

setup(
    name='hello-wordsmith',
    version='0.1.0',
    description='A simple Python package to interface with llama-index RAG over wordsmith data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://huggingface.co/datasets/derek-at-work/test/source env/bin/activate',
    author='Derek Johnston',
    author_email='derek@wordsmith.ai',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "llama-index",
        "chromadb",
        "llama-index-vector-stores-chroma"
    ],
    classifiers=[
        'Intended Audience :: End Users',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.6',
)
