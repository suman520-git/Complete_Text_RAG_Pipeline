import json
from pathlib import Path

from retrieval import retrieve

# Short names for our source files.
PRODUCT = "product_guide.txt"
SECURITY = "security_policy.txt"
OPERATIONS = "operations_runbook.txt"

# Each evidence item contains:
# (required source filename, required text excerpt)
#
# Empty evidence means the answer does not exist in our dataset.
questions = [
    {
        "id": 1,
        "category": "direct",
        "query": "How long are document backups retained?",
        "evidence": [
            (OPERATIONS, "Backups are retained for 14 days.")
        ]
    },
    {
        "id": 2,
        "category": "direct",
        "query": "How long are audit logs retained?",
        "evidence": [
            (SECURITY, "Audit logs are retained for 180 days.")
        ]
    },
    {
        "id": 3,
        "category": "direct",
        "query": "How many members and how much storage does Starter support?",
        "evidence": [
            (PRODUCT, "The Starter plan supports 10 members and provides 5 GB of storage.")
        ]
    },
    {
        "id": 4,
        "category": "direct",
        "query": "What is the maximum document size accepted by the API?",
        "evidence": [
            (PRODUCT, "The public API accepts a maximum document size of 2 MB.")
        ]
    },
    {
        "id": 5,
        "category": "direct",
        "query": "How often is a restore drill performed?",
        "evidence": [
            (OPERATIONS, "A restore drill is performed once every month.")
        ]
    },
    {
        "id": 6,
        "category": "conceptual",
        "query": "How does semantic search differ from keyword search?",
        "evidence": [
            (PRODUCT, "Keyword search looks for words that appear in the stored documents."),
            (PRODUCT, "Semantic search compares the meaning of a question with the meaning of document passages.")
        ]
    },
    {
        "id": 7,
        "category": "conceptual",
        "query": "What is the difference between authentication and authorization?",
        "evidence": [
            (SECURITY, "Authentication checks who a person is."),
            (SECURITY, "Authorization determines what that person is allowed to do.")
        ]
    },
    {
        "id": 8,
        "category": "conceptual",
        "query": "How does a readiness check differ from a liveness check?",
        "evidence": [
            (OPERATIONS, "A liveness check verifies that the application process is running."),
            (OPERATIONS, "A readiness check verifies that the application can accept requests and reach required dependencies.")
        ]
    },
    {
        "id": 9,
        "category": "conceptual",
        "query": "Why does a successful backup job not prove recovery will work?",
        "evidence": [
            (OPERATIONS, "A successful backup job does not prove that restoration will succeed."),
            (OPERATIONS, "During the monthly restore drill, operators check document contents and confirm that recovered documents can be found through search.")
        ]
    },
    {
        "id": 10,
        "category": "multi_document",
        "query": "What are the retention periods for backups and audit logs?",
        "evidence": [
            (OPERATIONS, "Backups are retained for 14 days."),
            (SECURITY, "Audit logs are retained for 180 days.")
        ]
    },
    {
        "id": 11,
        "category": "multi_document",
        "query": "What is the API request limit and how should clients handle HTTP 429?",
        "evidence": [
            (PRODUCT, "The request limit is 60 requests per minute per workspace."),
            (OPERATIONS, "the client should honor the Retry-After response header."),
            (OPERATIONS, "the client should use exponential backoff with jitter.")
        ]
    },
    {
        "id": 12,
        "category": "multi_document",
        "query": "Does the Team plan include single sign-on, and why is it useful?",
        "evidence": [
            (PRODUCT, "The Team plan includes semantic search and single sign-on."),
            (SECURITY, "It centralizes account access and makes employee offboarding easier.")
        ]
    },
    {
        "id": 13,
        "category": "unanswerable",
        "query": "What is the monthly price of the Team plan?",
        "evidence": []
    },
    {
        "id": 14,
        "category": "unanswerable",
        "query": "Who is the CEO of Nimbus Notes?",
        "evidence": []
    },
    {
        "id": 15,
        "category": "unanswerable",
        "query": "Which AWS region hosts Nimbus Notes?",
        "evidence": []
    }
]


def normalize_text(text):
    # Ignore differences in capitalization and whitespace.
    return " ".join(text.casefold().split())


def run_tests():

    project_folder = Path(__file__).resolve().parent
    reports_folder = project_folder / "reports"
    reports_folder.mkdir(exist_ok=True)

    # Validate our evidence labels against the saved source chunks.
    # This prevents a typo in a label from becoming a false failure.
    saved_path = project_folder / "storage" / "embeddings_500.json"
    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))

    source_texts = {}

    for record in saved_data["records"]:
        source = record["metadata"]["source"]
        source_texts.setdefault(source, "")
        source_texts[source] += "\n" + record["text"]

    for question in questions:
        for source, phrase in question["evidence"]:
            if normalize_text(phrase) not in normalize_text(source_texts[source]):
                raise ValueError(
                    f"Check evidence label for question {question['id']}: {phrase}"
                )

    all_results = []

    print("\n--- Retrieval evaluation ---")

    for chunk_size in [500, 1000]:

        for question in questions:

            results = retrieve(
                query=question["query"],
                top_k=5,
                chunk_size=chunk_size
            )

            missing_evidence = []
            evidence_found = 0

            for expected_source, expected_text in question["evidence"]:

                found = False

                for result in results:

                    source_matches = (
                        result["metadata"]["source"] == expected_source
                    )

                    text_matches = (
                        normalize_text(expected_text)
                        in normalize_text(result["text"])
                    )

                    if source_matches and text_matches:
                        found = True
                        break

                if found:
                    evidence_found += 1
                else:
                    missing_evidence.append({
                        "source": expected_source,
                        "text": expected_text
                    })

            required_count = len(question["evidence"])

            # Unknown-answer questions have no recall denominator.
            evidence_recall = None

            if required_count > 0:
                evidence_recall = evidence_found / required_count

            row = {
                "id": question["id"],
                "category": question["category"],
                "query": question["query"],
                "chunk_size": chunk_size,
                "top_k": 5,
                "evidence_recall": evidence_recall,
                "missing_evidence": missing_evidence,
                "context_characters": sum(
                    len(result["text"]) for result in results
                ),
                "results": results
            }

            all_results.append(row)

            print(
                f"\nQ{question['id']} | Size: {chunk_size} | "
                f"Evidence recall: {evidence_recall}"
            )
            print("Query:", question["query"])

            for rank, result in enumerate(results, start=1):
                metadata = result["metadata"]

                print(
                    f"\nRank {rank} | Source: {metadata['source']} | "
                    f"Page: N/A | Chunk: {metadata['chunk_id']} | "
                    f"Similarity: {result['similarity_score']:.4f}"
                )
                print(result["text"])

    output_path = reports_folder / "retrieval_results.json"

    output_path.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8"
    )

    print("\nSaved:", output_path)


# Run evaluation only when this file is executed directly.
if __name__ == "__main__":
    run_tests()