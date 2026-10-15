import re
import unicodedata

from load_documents import documents

# Store the cleaned documents in a separate list.
cleaned_documents = []

for document in documents:

    # Get the original document text.
    original_text = document["text"]

    # Normalize equivalent Unicode representations.
    normalized_text = unicodedata.normalize("NFC", original_text)

    # Split the text into individual lines.
    lines = normalized_text.splitlines()

    # Collect the non-empty, cleaned lines.
    cleaned_lines = []

    for line in lines:

        # Replace repeated spaces and tabs with one space.
        line = re.sub(r"[ \t]+", " ", line)

        # Remove spaces from the beginning and end.
        line = line.strip()

        # Keep only lines that still contain text.
        if line:
            cleaned_lines.append(line)

    # Join the remaining lines with one newline between them.
    cleaned_text = "\n".join(cleaned_lines)

    # Keep the source metadata with the cleaned text.
    cleaned_document = {
        "text": cleaned_text,
        "metadata": document["metadata"].copy()
    }

    cleaned_documents.append(cleaned_document)

    # Inspect the changes.
    print("\nSource:", document["metadata"]["source"])
    print("Characters before:", len(original_text))
    print("Characters after:", len(cleaned_text))
    print("Characters removed:", len(original_text) - len(cleaned_text))

print("\nTotal cleaned documents:", len(cleaned_documents))

# Display one complete cleaned document.
print("\nFirst cleaned document:")
print(cleaned_documents[0]["text"])