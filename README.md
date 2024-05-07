# Hello Wordsmith

This is the **Hello Wordsmith** package. This is a simple wrapper around the `llama-index CLI` project with some opinionated defaults. We aim to provide a "Hello World" experience using Retrieval-Augmented Generation (RAG).

For detailed information about the `llamaindex-rag` project, visit the [official documentation](https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/).

## Installation

Follow these steps to install and set up your environment:

**Setup**:
1. `pip install git+https://github.com/wordsmith-ai/hello-wordsmith -q`
2. `export OPENAI_API_KEY="sk-..."`

**Usage**:
1. `hello-wordsmith` # Launch an interactive chat.
2. `hello-wordsmith -q 'What is article III about?'` # Single question and answer
3. `hello-wordsmith -f "./my_directory/*"` # Ingest and index your own data to query
4. `hello-wordsmith --clear` # Clear stored data
