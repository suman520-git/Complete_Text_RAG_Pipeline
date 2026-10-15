import json
from pathlib import Path

import faiss
import numpy as np

project_folder = Path(__file__).resolve().parent
storage_folder = project_folder / "storage"

print("\n--- Building FAISS indexes ---")

for chunk_size in [500, 1000]:

    # Read the embeddings already saved in Step 13.
    input_path = storage_folder / f"embeddings_{chunk_size}.json"

    saved_data = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    records = saved_data["records"]

    # Extract the vectors in their original order.
    vectors = []

    for record in records:
        vectors.append(record["vector"])

    # FAISS expects a two-dimensional array of float32 numbers.
    vector_array = np.array(vectors, dtype="float32")

    # Each row is one vector; each column is one dimension.
    dimensions = vector_array.shape[1]

    # Normalize each vector to length 1.
    faiss.normalize_L2(vector_array)

    # Create an index that compares vectors using inner product.
    index = faiss.IndexFlatIP(dimensions)

    # Add all vectors to the index.
    index.add(vector_array)

    # Save the index to disk.
    index_path = storage_folder / f"index_{chunk_size}.faiss"
    faiss.write_index(index, str(index_path))

    # Reload it to verify that it was saved successfully.
    loaded_index = faiss.read_index(str(index_path))

    print("\nChunk size:", chunk_size)
    print("Vector array shape:", vector_array.shape)
    print("Stored vectors:", loaded_index.ntotal)
    print("Dimensions:", loaded_index.d)
    print("Saved index:", index_path.name)