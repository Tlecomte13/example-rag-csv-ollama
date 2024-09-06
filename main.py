import argparse
import time

import yaml
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

from helper.get_embedding import get_embedding

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # CLI
    query_text = input("Please enter your question: ")

    query_rag(query_text)


def query_rag(query_text: str):
    embedding_function = get_embedding()
    db = Chroma(persist_directory=config["chroma_path"], embedding_function=embedding_function)

    with tqdm(total=100, desc="Processing") as pbar:
        time.sleep(1)
        pbar.update(20)

        results = db.similarity_search_with_score(query_text, k=5)
        pbar.update(40)

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        pbar.update(20)

        model = Ollama(model=config["llm_model"])
        response_text = model.invoke(prompt)
        pbar.update(20)

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    console = Console()
    console.print(Panel(f"\n{response_text}\n", title="Response"))
    console.print(Panel(f"{sources}", title="sources"))

    return response_text


if __name__ == "__main__":
    main()
