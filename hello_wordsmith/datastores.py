import os

import chromadb
from llama_index.cli.rag import default_ragcli_persist_dir
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic.v1 import BaseModel


class InitialisedDataContainer(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    db: chromadb.Collection
    doc_store: SimpleDocumentStore
    vector_store: ChromaVectorStore
    index: VectorStoreIndex
    storage_context: StorageContext


def _get_chroma_db() -> chromadb.Collection:
    db = chromadb.PersistentClient(
        path=os.path.join(default_ragcli_persist_dir(), "chroma")
    )
    chroma_collection = db.get_or_create_collection("wordsmith_rag_demo_index")
    return chroma_collection


def fetch_or_initialise_datastores() -> InitialisedDataContainer:
    db = _get_chroma_db()
    vector_store = ChromaVectorStore(chroma_collection=db)
    docstore = SimpleDocumentStore()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, docstore=docstore,
    )
    if not db.count():
        package_directory = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(package_directory, "public_wordsmith_dataset")
        docs = SimpleDirectoryReader(
            input_dir=dataset_path, filename_as_id=True
        ).load_data()
        docstore.add_documents(docs)
        index = VectorStoreIndex.from_documents(
            documents=docs, storage_context=storage_context
        )
        storage_context.persist(persist_dir=default_ragcli_persist_dir())
    else:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return InitialisedDataContainer(
        db=db,
        doc_store=docstore,
        vector_store=vector_store,
        index=index,
        storage_context=storage_context,
    )
