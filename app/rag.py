import sys

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import MIN_RELEVANCE_SCORE
from app.llm import get_llm
from app.vectorstore import get_vectorstore

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY the provided context.
Rules:
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
- Cite every claim with the source tag it came from, e.g. [1706.03762v7.pdf p.5].
- Be concise."""

NO_ANSWER = "I don't know based on the provided documents."

def retrieve(question: str, k: int = 4, min_score:float = MIN_RELEVANCE_SCORE) -> list[Document]:
    results = get_vectorstore().similarity_search_with_relevance_scores(question, k=k)
    docs = []
    for doc, score in results:
        if score >= min_score:
            doc.metadata["score"] = round(score, 3)
            docs.append(doc)
    return docs

def format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        m = doc.metadata
        parts.append(f"[{m['source']} p.{m['page']}]\n{doc.page_content}")
    return "\n\n".join(parts)


def answer(question: str, provider: str = "openai", k: int = 4) -> str:
    docs = retrieve(question, k=k)
    if not docs:
        return {"answer": NO_ANSWER, "sources": []}
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{format_context(docs)}\n\nQuestion: {question}"),
    ]
    response = get_llm(provider).invoke(messages)
    return {"answer": response.text, "sources": [doc.metadata for doc in docs]}



if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How does multi-head attention work?"
    for provider in ["openai", "anthropic", "gemini"]:
        result = answer(question, provider=provider)
        print(f"===== {provider} =====")
        print(result["answer"])
        for s in result["sources"]:
            print(f"  {s['score']:.3f}  {s['source']} p.{s['page']}")
        print()