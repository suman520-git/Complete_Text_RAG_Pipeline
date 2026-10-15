from langchain_text_splitters import RecursiveCharacterTextSplitter

from clean_documents import cleaned_documents

# Compare these two maximum chunk lengths.
chunk_sizes = [500, 1000]

# Store each strategy's chunks separately.
chunks_by_size = {}

print("\n--- Chunking results ---")

for chunk_size in chunk_sizes:

    # Create a splitter for the current chunk size.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Collect chunks from all three documents for this strategy.
    all_chunks = []

    print("\nChunk size:", chunk_size)

    for document in cleaned_documents:

        # Copy the original source information.
        metadata = document["metadata"].copy()

        # TXT documents do not have page numbers.
        metadata["page"] = None

        # Split this document while preserving its metadata.
        document_chunks = splitter.create_documents(
            texts=[document["text"]],
            metadatas=[metadata]
        )

        # Number the chunks within this source document.
        for chunk_number, chunk in enumerate(document_chunks, start=1):

            chunk.metadata["chunk_id"] = chunk_number
            chunk.metadata["chunk_size"] = chunk_size

            all_chunks.append(chunk)

        print(
            "Source:",
            metadata["source"],
            "| Chunks:",
            len(document_chunks)
        )

    # Save this strategy's complete chunk list.
    chunks_by_size[chunk_size] = all_chunks

    print("Total chunks:", len(all_chunks))

    # Verify that every chunk fits the configured maximum.
    within_limit = all(
        len(chunk.page_content) <= chunk_size
        for chunk in all_chunks
    )

    print("All chunks within size limit:", within_limit)

    # Inspect the first two chunks.
    for chunk in all_chunks[:2]:

        print("\nMetadata:", chunk.metadata)
        print("Actual characters:", len(chunk.page_content))
        print("Text:")
        print(chunk.page_content)