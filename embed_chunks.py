import json
from pathlib import Path

from embedding import embedding_model, embedding_model_name
from chunk_documents import chunks_by_size

# Create a folder for our saved embeddings and indexes.
project_folder = Path(__file__).resolve().parent
storage_folder = project_folder / "storage"
storage_folder.mkdir(exist_ok=True)

print("\n--- Embedding all chunks ---")

for chunk_size, chunks in chunks_by_size.items():

    # Collect the text from every chunk in this strategy.
    texts = []

    for chunk in chunks:
        texts.append(chunk.page_content)

    # Request one embedding for each chunk.
    vectors = embedding_model.embed_documents(texts)

    # Check that every chunk received an embedding.
    if len(vectors) != len(chunks):
        raise ValueError("The chunk count and embedding count do not match.")

    # Keep each chunk's text, metadata and vector together.
    records = []

    for chunk, vector in zip(chunks, vectors):

        record = {
            "text": chunk.page_content,
            "metadata": chunk.metadata,
            "vector": vector
        }

        records.append(record)

    # Include the model name so we can check it during retrieval.
    saved_data = {
        "model": embedding_model_name,
        "chunk_size": chunk_size,
        "dimensions": len(vectors[0]),
        "records": records
    }

    # Save each strategy in a separate JSON file.
    output_path = storage_folder / f"embeddings_{chunk_size}.json"

    output_path.write_text(
        json.dumps(saved_data, indent=2),
        encoding="utf-8"
    )

    print("\nChunk size:", chunk_size)
    print("Chunks:", len(chunks))
    print("Embeddings:", len(vectors))
    print("Dimensions:", len(vectors[0]))
    print("Saved file:", output_path.name)