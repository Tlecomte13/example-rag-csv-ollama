import yaml
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def get_embedding():
    embeddings = OllamaEmbeddings(model=config["embeddings_model"])
    return embeddings
