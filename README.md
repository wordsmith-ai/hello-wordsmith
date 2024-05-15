# Hello Wordsmith

This is the **Hello Wordsmith** package. This is a simple wrapper around the `llama-index CLI` project with some opinionated defaults. We aim to provide a "Hello World" experience using Retrieval-Augmented Generation (RAG). For more context on what RAG is, tradeoffs and, and a detailed walthrough of this project, see [this The Pragmatic Engineer article](https://newsletter.pragmaticengineer.com/p/rag). 

For detailed information about the `llamaindex-rag` project, visit the [official documentation](https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/).

## Pre-requisites for usage

An active OpenAI subscription. Ensure you are registered with OpenAI, your [billing details added](https://platform.openai.com/settings/organization/billing/overview) and have an [API key to use](https://platform.openai.com/api-keys)

## Installation

Follow these steps to install and set up your environment:

**Setup**:
1. `pip install git+https://github.com/wordsmith-ai/hello-wordsmith -q`
2. `export OPENAI_API_KEY="sk-..."`

Note:\
It's best practice to work in a virtual Python environment, as opposed to your system's default  Python installation. Popular solutions include `venv`, `conda`, and `pipenv`. If you *do* use
your system Python, make sure the bin dir is on your PATH, e.g. `export PATH="/Library/Frameworks/Python.framework/Versions/3.x/bin:${PATH}`

**Use**:
1. `hello-wordsmith` // Launch an interactive chat.
2. `hello-wordsmith -q 'What is article III about?'` // Single question and answer
3. `hello-wordsmith -f "./my_directory/*" --chunk-size 256 --chunk-overlap 128` // Ingest and index your own data to query with custom document chunk sizes and overlaps
4. `hello-wordsmith --clear` // Clear stored data

**Example installation and usage via venv**

Using Python 3, on a Mac:

1. `python3 -m venv hello-wordsmith` // Initialize the venv virtual environment folder
2. `cd hello-wordsmith`
3. `source ./bin/activate` // Launch the virtual environment
4. `pip install git+https://github.com/wordsmith-ai/hello-wordsmith -q` // Install the hello-wordsmith package, suppressing output with the -q flag. Remove this flag to see install progress
5. `export OPENAI_API_KEY="sk-..."` // Export your OpenAI key
6. `hello-wordsmith -q 'What is article III about?'` // Send a single question, and wait for the answer to arrive using the RAG
7. `hello-wordsmith --chunk-size 256 --chunk-overlap 64` // Start the interactive assistant to ask questions and answers:

<img width="509" alt="example" src="https://github.com/wordsmith-ai/hello-wordsmith/assets/1094502/beb3df38-734f-49b0-9d46-5d6386779e71">

**Explore**:\
As you can see, this repo is an extremely simplistic first step towards building a RAG system on your data. You can open up these files and explore how changing parameters like chunk size, or the 
embedding model that we use, can influence the performance of the system.
