import json
from pathlib import Path
from statistics import mean

project_folder = Path(__file__).resolve().parent

results_path = (
    project_folder / "reports" / "retrieval_results.json"
)

rows = json.loads(
    results_path.read_text(encoding="utf-8")
)

average_scores = {}

print("\n--- Chunking comparison ---")

for chunk_size in [500, 1000]:

    # Only answerable questions belong in the recall calculation.
    answerable_rows = [
        row for row in rows
        if row["chunk_size"] == chunk_size
        and row["evidence_recall"] is not None
    ]

    average_recall = mean(
        row["evidence_recall"] for row in answerable_rows
    )

    complete_count = sum(
        row["evidence_recall"] == 1.0
        for row in answerable_rows
    )

    average_context = mean(
        row["context_characters"] for row in answerable_rows
    )

    average_scores[chunk_size] = average_recall

    print("\nChunk size:", chunk_size)
    print("Answerable questions:", len(answerable_rows))
    print("Mean evidence recall:", round(average_recall, 4))
    print("Questions with all evidence:", complete_count)
    print("Average retrieved characters:", round(average_context))

    print("Questions with missing evidence:")

    failures = [
        row for row in answerable_rows
        if row["evidence_recall"] < 1.0
    ]

    if not failures:
        print("None in this test set.")

    for row in failures:
        print(f"Q{row['id']}: {row['query']}")

        for evidence in row["missing_evidence"]:
            print("  Missing source:", evidence["source"])
            print("  Missing text:", evidence["text"])

score_500 = average_scores[500]
score_1000 = average_scores[1000]

if abs(score_500 - score_1000) < 0.000001:
    print("\nResult: Both strategies tied on mean evidence recall.")
elif score_500 > score_1000:
    print("\nResult: Size 500 achieved higher mean evidence recall.")
else:
    print("\nResult: Size 1000 achieved higher mean evidence recall.")

print("\n--- Missing-answer example ---")

for row in rows:
    if row["id"] == 13:

        top_result = row["results"][0]

        print("\nChunk size:", row["chunk_size"])
        print("Question:", row["query"])
        print("Top source:", top_result["metadata"]["source"])
        print(
            "Top similarity:",
            round(top_result["similarity_score"], 4)
        )
        print("The dataset contains no monthly Team price.")
        print("Related passages cannot supply the missing price.")