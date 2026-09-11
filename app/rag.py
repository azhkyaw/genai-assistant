import sys

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.vectorstore import get_vectorstore

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY the provided context.
Rules:
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
- Cite every claim with the source tag it came from, e.g. [1706.03762v7.pdf p.5].
- Be concise."""


def retrieve(question: str, k: int = 4) -> list[Document]:
    return get_vectorstore().similarity_search(question, k=k)


def format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(f"[{m['source']} p.{m['page']}]\n{doc.page_content}")
    return "\n\n".join(parts)


def answer(question: str, provider: str = "openai", k: int = 4) -> str:
    docs = retrieve(question, k=k)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{format_context(docs)}\n\nQuestion: {question}"),
    ]
    response = get_llm(provider).invoke(messages)
    return response.text


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How does multi-head attention work?"
    for provider in ["openai", "anthropic", "gemini"]:
        print(f"===== {provider} =====")
        print(answer(question, provider=provider))
        print()