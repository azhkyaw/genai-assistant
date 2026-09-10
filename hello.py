from app.llm import get_llm

QUESTION = "In one sentence, what is retrieval-augmented generation"

for provider in ["openai", "anthropic", "gemini"]:
    llm = get_llm(provider)
    response = llm.invoke(QUESTION)
    print(f"[{provider}] {response.text}\n")