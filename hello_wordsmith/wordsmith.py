import os
import sys

import chromadb
from llama_index.cli.rag import RagCLI
from llama_index.core import (
    ChatPromptTemplate,
    SimpleDirectoryReader,
    VectorStoreIndex
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_pipeline import InputComponent, QueryPipeline
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import Settings
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
    OpenAIEmbeddingModelType
)
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)


def initialize_chroma_db():
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("wordsmith")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


def setup_document_storage(vector_store):
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(package_directory, "public_wordsmith_dataset")
    reader = SimpleDirectoryReader(input_dir=dataset_path)
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index


def initialize_llm():
    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4")
    return llm


_system_prompt = ChatMessage(
    content=(
        "You are an expert Q&A analyst representing Wordsmith in front of "
        "potentially interested users.\n"
        "If the question is related to Wordsmith in any way, "
        "answer the query using the provided context information.\n"
        "If you can't find the answer in the provided context information, "
        "simply say you don't have enough information to answer the query.\n"
        "Always be polite and professional.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...', etc."
    ),
    role=MessageRole.SYSTEM,
)

_chat_template_messages = [
    _system_prompt,
    ChatMessage(
        content=(
            "Context information from multiple sources is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]


def configure_query_pipeline(index, llm):
    """Configure and set up the query pipeline"""
    text_qa_chat_template = ChatPromptTemplate.from_messages(
        _chat_template_messages
    )
    query_pipeline = QueryPipeline()

    retriever = index.as_retriever(similarity_top_k=20)
    summarizer = TreeSummarize(
        llm=llm, streaming=True, summary_template=text_qa_chat_template
    )

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
