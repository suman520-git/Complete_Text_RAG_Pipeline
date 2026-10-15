from pathlib import Path

# Find the folder containing this Python file.
project_folder = Path(__file__).resolve().parent

# Build the path to our text document.
document_path = project_folder / "data" / "product_guide.txt"

# Read the document as a Python string.
text = document_path.read_text(encoding="utf-8")

# Inspect what we loaded.
print("Document:", document_path.name)
print("Python data type:", type(text))

print("\nDocument content:")
print(text)

