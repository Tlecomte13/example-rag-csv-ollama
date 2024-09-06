# Example Project: create RAG (Retrieval-Augmented Generation) with LangChain and Ollama

This project uses LangChain to load CSV documents, split them into chunks, store them in a Chroma database, and query this database using a language model.

## Prerequisites

- [Ollama](https://ollama.com/download)
- Python 3.8 or higher
- pip

## Installation

1. Clone the repository:
    ```bash
    git clone <REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    ```
   
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
   
4. Install model with Ollama:
    ```bash
    ollama pull <YOUR_MODEL> 
    ```

## Configuration

Ensure that the `config.yaml` file is correctly configured.

## Usage
Add documents to the database
To add documents to the Chroma database, run:

```bash
python add_csv_in_database.py
```

You can reset the database using the --reset option:

```bash
python add_csv_in_database.py --reset
```

Query the database

To query the database, run:

```bash
python main.py
```

## Project Structure
* **add_csv_in_database.py**: Script to load CSV documents, split them into chunks, and add them to the Chroma database.
* **main.py**: Script to query the Chroma database and generate context-based responses.
* **helper/get_embedding_function.py**: Utility function to get the embedding function.
* **config.yaml**: Configuration file for file paths, models, and text splitting parameters.

## Dependencies

* langchain_community
* langchain_chroma
* tqdm
* rich
* pyyaml

### Sources

https://www.sakunaharinda.xyz/ragatouille-book/intro.html  
https://ollama.com/  
https://www.youtube.com/watch?v=2TJxpyO3ei4  
https://python.langchain.com/v0.2/docs/introduction/
