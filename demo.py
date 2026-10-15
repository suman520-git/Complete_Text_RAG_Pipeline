import json
from pathlib import Path

from rag import answer_question
from test_queries import questions

# Include direct, conceptual, multi-document and unknown questions.
selected_ids = {1, 2, 3, 6, 8, 10, 11, 13}

demo_results = []

print("\n--- Eight-query demonstration ---")

for question in questions:

    if question["id"] not in selected_ids:
        continue

    result = answer_question(
        query=question["query"],
        top_k=5,
        chunk_size=500
    )

    # Include the question ID and category in the saved report.
    result["question_id"] = question["id"]
    result["category"] = question["category"]

    demo_results.append(result)

    print("\nQuestion ID:", question["id"])
    print("Question:", result["query"])
    print("Answer:", result["answer"])

    print("Retrieved sources:")

    for source in result["sources"]:

        metadata = source["metadata"]

        print(
            f"[{source['citation']}] "
            f"{metadata['source']} | "
            f"Chunk {metadata['chunk_id']} | "
            f"Similarity {source['similarity_score']:.4f}"
        )

# Save complete results, including all retrieved passage text.
project_folder = Path(__file__).resolve().parent
reports_folder = project_folder / "reports"
reports_folder.mkdir(exist_ok=True)

output_path = reports_folder / "eight_query_demo.json"

output_path.write_text(
    json.dumps(demo_results, indent=2),
    encoding="utf-8"
)

print("\nDemonstrated questions:", len(demo_results))
print("Saved:", output_path)