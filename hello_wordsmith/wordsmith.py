from argparse import ArgumentParser
import os
import sys
from typing import Callable, Union

from llama_index.cli.rag import RagCLI
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.llms.openai import OpenAI

from .datastores import fetch_or_initialise_datastores
from .query_pipeline import configure_query_pipeline


def _add_chunk_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Document chunk size for embedding generation.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Document chunk overlap for embedding generation.",
    )
    return parser


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

    @classmethod
    def add_parser_args(
        cls,
        parser: ArgumentParser,
        instance_generator: Union[Callable[[], RagCLI], None]
    ) -> None:
        _add_chunk_args(parser)
        super().add_parser_args(parser, instance_generator)


def _init_env(func: Callable[[], None]) -> Callable[[], None]:
    def wrapper() -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(
                "Error: Environment variable 'OPENAI_API_KEY' is not set. Please set this before running."
            )
            sys.exit(1)
        Settings.embed_model = OpenAIEmbedding(
            model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
        )
        return func()

    return wrapper


def _set_chunking_settings() -> None:
    parser = ArgumentParser(description="Args for document chunking.")
    _add_chunk_args(parser)
    args, _ = parser.parse_known_args()
    Settings.chunk_size = args.chunk_size
    Settings.chunk_overlap = args.chunk_overlap


@_init_env
def main() -> None:
    _set_chunking_settings()
    datastore_container = fetch_or_initialise_datastores()
    llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4")
    query_pipeline = configure_query_pipeline(
        index=datastore_container.index,
        llm=llm
    )
    ingestion_pipeline = IngestionPipeline(
        vector_store=datastore_container.vector_store,
        cache=IngestionCache(),
        docstore=datastore_container.doc_store,
    )
    rag_cli_instance = WordsmithRAGCLI(
        ingestion_pipeline=ingestion_pipeline,
        llm=llm,
        query_pipeline=query_pipeline
    )
    rag_cli_instance.cli()


if __name__ == "__main__":
    main()
