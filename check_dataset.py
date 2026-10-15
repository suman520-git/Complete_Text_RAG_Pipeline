from clean_documents import cleaned_documents

print("\n--- Expanded dataset check ---")

total_characters = 0

for document in cleaned_documents:

    # Read this document's source and cleaned text length.
    source = document["metadata"]["source"]
    character_count = len(document["text"])

    # Add its length to the running total.
    total_characters += character_count

    print("\nSource:", source)
    print("Cleaned characters:", character_count)
    print("Longer than 1000 characters:", character_count > 1000)

print("\nTotal documents:", len(cleaned_documents))
print("Total cleaned characters:", total_characters)