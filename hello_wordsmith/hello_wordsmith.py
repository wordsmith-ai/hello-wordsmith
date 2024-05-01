#!/path/to/your/virtualenv/bin/python
import os
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import SimpleDirectoryReader
from llama_index.cli.rag import RagCLI
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


# optional, set any API keys your script may need (perhaps using python-dotenv library instead)
# os.environ["OPENAI_API_KEY"] = "sk-xxx"

from llama_index.core import VectorStoreIndex, StorageContext

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("wordsmith")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


package_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(package_directory, 'public_wordsmith_dataset')
reader = SimpleDirectoryReader(input_dir=dataset_path)
docs = reader.load_data()
index = VectorStoreIndex.from_documents(
    docs, storage_context=storage_context
)

# docstore = SimpleDocumentStore()

llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4")

custom_ingestion_pipeline = IngestionPipeline(
    vector_store=vector_store,
)

from llama_index.core import PromptTemplate


prompt_str = "Please generate related movies to {query_str}"
prompt_tmpl = PromptTemplate(prompt_str)
query_pipeline = QueryPipeline(verbose=True)

from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.query_pipeline import InputComponent

# construct vector store and customize storage context

retriever = index.as_retriever(similarity_top_k=5)
summarizer = TreeSummarize(llm=llm)
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

# you can optionally specify your own custom readers to support additional file types.
# file_extractor = {".html": ...}

rag_cli_instance = RagCLI(
    ingestion_pipeline=custom_ingestion_pipeline,
    llm=llm,
    query_pipeline=query_pipeline
)


def main():
    rag_cli_instance.cli()


if __name__ == "__main__":
    main()
