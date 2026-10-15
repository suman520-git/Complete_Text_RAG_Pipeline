from rag import answer_question

# Test a direct fact, a question using two documents,
# and a question whose answer is missing.
questions = [
    "How long are document backups retained?",
    "What are the retention periods for backups and audit logs?",
    "What is the monthly price of the Team plan?"
]

print("\n--- Answer generation check ---")

for question in questions:

    result = answer_question(
        query=question,
        top_k=5,
        chunk_size=500
    )

    print("\nQuestion:", result["query"])
    print("Answer:", result["answer"])

    print("\nRetrieved sources:")

    for source in result["sources"]:

        metadata = source["metadata"]

        print(
            f"[{source['citation']}] "
            f"{metadata['source']} | "
            f"Chunk {metadata['chunk_id']} | "
            f"Similarity {source['similarity_score']:.4f}"
        )