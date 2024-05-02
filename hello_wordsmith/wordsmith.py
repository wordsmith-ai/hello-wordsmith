from argparse import ArgumentParser
import os
import sys

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    PromptTemplate,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_pipeline import QueryPipeline, InputComponent
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.cli.rag import RagCLI
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


def initialize_chroma_db():
    """Initialize the ChromaDB client and collection"""
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("wordsmith")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


def setup_document_storage(vector_store):
    """Set up document storage and load data"""
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(package_directory, "public_wordsmith_dataset")
    reader = SimpleDirectoryReader(input_dir=dataset_path)
    docs = reader.load_data()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    return index


def initialize_llm():
    """Initialize the Large Language Model"""
    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4")
    return llm


def configure_query_pipeline(index, llm):
    """Configure and set up the query pipeline"""
    prompt_str = "Please generate related movies to {query_str}"
    prompt_tmpl = PromptTemplate(prompt_str)
    query_pipeline = QueryPipeline()

    retriever = index.as_retriever(similarity_top_k=5)
    summarizer = TreeSummarize(llm=llm, streaming=True)

    query_pipeline.add_modules(
        {
            "input": InputComponent(),
            "retriever": retriever,
            "summarizer": summarizer,
        }
    )
    query_pipeline.add_link("input", "retriever")
    query_pipeline.add_link("input", "summarizer", dest_key="query_str")
    query_pipeline.add_link("retriever", "summarizer", dest_key="nodes")

    return query_pipeline


class WordsmithRAGCLI(RagCLI):

    def cli(self) -> None:
        """
        Entrypoint for CLI tool.
        """
        if len(sys.argv) == 1:
            sys.argv.extend(["rag", "-c"])
        elif "rag" not in sys.argv:
            sys.argv.insert(1, "rag")
        super().cli()


def main():
    vector_store = initialize_chroma_db()
    index = setup_document_storage(vector_store)
    llm = initialize_llm()
    query_pipeline = configure_query_pipeline(index, llm)
    ingestion_pipeline = IngestionPipeline(vector_store=vector_store)
    rag_cli_instance = WordsmithRAGCLI(
        ingestion_pipeline=ingestion_pipeline,
        llm=llm,
        query_pipeline=query_pipeline
    )
    rag_cli_instance.cli()


if __name__ == "__main__":
    main()
