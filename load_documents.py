from pathlib import Path

# Find the folder containing this Python file.
project_folder = Path(__file__).resolve().parent

# Locate the folder containing our documents.
data_folder = project_folder / "data"

# Find all TXT files and sort them by filename.
file_paths = sorted(data_folder.glob("*.txt"))

# Start with an empty list to hold our loaded documents.
documents = []

# Read one file at a time.
for file_path in file_paths:

    # Read the file content as a string.
    text = file_path.read_text(encoding="utf-8")

    # Keep the text and its source filename together.
    document = {"text": text,
        "metadata": {
            "source": file_path.name,
            "type": "text"
        }
    }

    # Add this document to the list.
    documents.append(document)

# Check how many documents were loaded.
print("Total documents loaded:", len(documents))

# Display a short summary of each document.
for document in documents:
    print("\nSource:", document["metadata"]["source"])
    print("Characters:", len(document["text"]))
    print("Preview:", document["text"][:100])