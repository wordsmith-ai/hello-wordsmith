from llama_index.core import ChatPromptTemplate, VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.query_pipeline import InputComponent, QueryPipeline
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.llms.openai import OpenAI

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

_TOP_K_RETRIEVAL = 20


def configure_query_pipeline(*, index: VectorStoreIndex, llm: OpenAI) -> QueryPipeline:
    """Configure and set up the query pipeline"""
    text_qa_chat_template = ChatPromptTemplate.from_messages(_chat_template_messages)
    query_pipeline = QueryPipeline()

    retriever = index.as_retriever(similarity_top_k=_TOP_K_RETRIEVAL)
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
