import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Find the project folder.
project_folder = Path(__file__).resolve().parent

# Load settings from the project's .env file.
load_dotenv(project_folder / ".env")

# Read the API key without printing it.
api_key = os.getenv("OPENAI_API_KEY")

# Stop early if the key has not been configured.
if not api_key or api_key == "replace_with_your_actual_api_key":
    raise ValueError("Set your actual OPENAI_API_KEY in the .env file.")

# Read the embedding model name.
embedding_model_name = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-ada-002"
)

# Configure the LangChain client used to request embeddings.
embedding_model = OpenAIEmbeddings(
    model=embedding_model_name,
    api_key=api_key,
    request_timeout=60,
    max_retries=2
)