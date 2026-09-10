from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DATA_DIR = Path("data")

def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Read every .pdf, .md and .txt file in data_dir into Document objects."""
    docs = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix == ".pdf":
            reader = PdfReader(path)
            for page_number, page in enumerate(reader.pages, start=1):
                docs.append(Document(
                    page_content=page.extract_text() or "",
                    metadata={"source": path.name, "page": page_number},
                ))
        elif path.suffix in (".md", ".txt"):
            docs.append(Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source: path.name"}
            ))
    return docs

def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")
    print("--- first chunk ---")
    print(chunks[0].page_content)
    print(chunks[0].metadata)