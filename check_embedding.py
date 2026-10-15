from embedding import embedding_model, embedding_model_name
from chunk_documents import chunks_by_size

# Select the first chunk from the 500-character strategy.
first_chunk = chunks_by_size[500][0]

# Extract its text.
text = first_chunk.page_content

print("\n--- Embedding check ---")

print("Model:", embedding_model_name)
print("Source:", first_chunk.metadata["source"])
print("Chunk ID:", first_chunk.metadata["chunk_id"])
print("Text characters:", len(text))

# Send a list containing one chunk to OpenAI.
# The result is a list containing one embedding vector.
vectors = embedding_model.embed_documents([text])

# Take the first and only vector.
vector = vectors[0]

print("\nNumber of embeddings:", len(vectors))
print("Vector data type:", type(vector))
print("Embedding dimensions:", len(vector))
print("First 5 values:", vector[:5])