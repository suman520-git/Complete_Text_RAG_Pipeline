from retrieval import retrieve

query = "How long are document backups retained?"

print("\n--- Retrieval check ---")
print("Query:", query)

# Search each chunking strategy with the same question.
for chunk_size in [500, 1000]:

    results = retrieve(
        query=query,
        top_k=5,
        chunk_size=chunk_size
    )

    print("\nChunk size:", chunk_size)
    print("Results returned:", len(results))

    for rank, result in enumerate(results, start=1):

        metadata = result["metadata"]

        # TXT files do not have page numbers.
        page = metadata.get("page")

        if page is None:
            page = "N/A"

        print("\nRank:", rank)
        print("Source:", metadata["source"])
        print("Page:", page)
        print("Chunk ID:", metadata["chunk_id"])

        print(
            "Cosine similarity:",
            round(result["similarity_score"], 4)
        )

        print("Retrieved text:")
        print(result["text"])