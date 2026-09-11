
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"}
    )

if __name__ == "__main__":
    vs = get_vectorstore()
    results = vs.similarity_search("How does multi-head attention work?", k=3)
    for doc in results:
        print(f"[{doc.metadata['source']} p.{doc.metadata.get('page')}]")
        print(doc.page_content[:300].replace("\n", " "))
        print()