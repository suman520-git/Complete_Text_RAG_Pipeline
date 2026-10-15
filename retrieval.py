import json
from pathlib import Path

import faiss
import numpy as np

from embedding import embedding_model, embedding_model_name

project_folder = Path(__file__).resolve().parent
storage_folder = project_folder / "storage"


def retrieve(query, top_k=5, chunk_size=500):

    # Remove spaces around the question.
    query = query.strip()

    # Validate the inputs before calling OpenAI.
    if not query:
        raise ValueError("The query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if chunk_size not in [500, 1000]:
        raise ValueError("chunk_size must be 500 or 1000.")

    # Locate the saved index and its matching records.
    index_path = storage_folder / f"index_{chunk_size}.faiss"
    records_path = storage_folder / f"embeddings_{chunk_size}.json"

    # Load the vector index.
    index = faiss.read_index(str(index_path))

    # Load the original text, metadata and model information.
    saved_data = json.loads(
        records_path.read_text(encoding="utf-8")
    )

    records = saved_data["records"]

    # Documents and queries must use the same embedding model.
    if saved_data["model"] != embedding_model_name:
        raise ValueError(
            "Embedding model changed. Rebuild embeddings and indexes."
        )

    # Check that the index and records have matching sizes.
    if index.ntotal != len(records):
        raise ValueError("The index and saved records do not match.")

    if index.d != saved_data["dimensions"]:
        raise ValueError("The saved embedding dimensions do not match.")

    # Ask OpenAI to embed this question.
    query_vector = embedding_model.embed_query(query)

    # Make a table with one row: our query vector.
    query_array = np.array([query_vector], dtype="float32")

    if query_array.shape[1] != index.d:
        raise ValueError("Query and document dimensions do not match.")

    # Normalize the query, just as we normalized the document vectors.
    faiss.normalize_L2(query_array)

    # Do not request more results than the index contains.
    number_of_results = min(top_k, index.ntotal)

    # Search for the most similar vectors.
    scores, positions = index.search(
        query_array,
        number_of_results
    )

    results = []

    # There is one row of results because we searched one question.
    for score, position in zip(scores[0], positions[0]):

        # FAISS uses -1 when no matching index entry is available.
        if position == -1:
            continue

        # Recover the text and metadata at this vector's position.
        record = records[int(position)]

        result = {
            "text": record["text"],
            "metadata": record["metadata"],
            "similarity_score": float(score)
        }

        results.append(result)

    return results