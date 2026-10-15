# Complete Text RAG Pipeline

Project folder: Complete_Text_RAG_Pipeline

## Objective

Build a text RAG pipeline using OpenAI, LangChain, FAISS and FastAPI.

RAG means Retrieval-Augmented Generation. The application retrieves relevant
document passages and supplies them to an LLM to generate an answer with citations.

## Dataset

Three fictional Nimbus Notes documents:

- product_guide.txt
- security_policy.txt
- operations_runbook.txt

The dataset contains 6,081 cleaned characters.

Preprocessing normalizes Unicode, removes empty lines, collapses repeated
spaces and tabs, and preserves source metadata.

## Chunking

LangChain RecursiveCharacterTextSplitter creates two strategies:

| Chunk size | Target overlap | Chunks |
|---|---:|---:|
| 500 characters | 100 characters | 15 |
| 1000 characters | 100 characters | 8 |

Actual overlap can be smaller because splitting respects text boundaries.

Every chunk contains source, type, page, chunk_id and chunk_size metadata.
Page is null because TXT documents have no page numbers.

## Embeddings

OpenAI text-embedding-ada-002 represents each chunk as a vector of 1,536 numbers.
This is the model used for the assignment's “OpenAI Embedding 2” requirement.

Embeddings allow retrieval by meaning rather than only exact word matches.
Similar meanings tend to produce nearby vectors, but similarity does not
guarantee that a passage answers a question.

Vector dimensions determine the number of values stored and compared.
Documents and questions must use the same embedding model and dimensions.

## Vector storage and retrieval

FAISS is a local vector-search library. We use IndexFlatIP with normalized
document and query vectors, making its scores cosine similarities.

JSON files preserve text, metadata, model information and original vectors.
FAISS index positions map to corresponding JSON records.

retrieve(query, top_k=5, chunk_size=500):

1. Loads the selected index and records.
2. Embeds the question using OpenAI.
3. Normalizes the query vector.
4. Searches for the top-k vectors.
5. Returns text, metadata and similarity scores.

Higher scores mean greater similarity, not a probability of correctness.

## Answer generation

The OpenAI chat model receives the question and retrieved passages.
It is instructed to answer from those passages, cite source labels and
acknowledge missing information.

Citations must still be reviewed against their supporting passages.
A prompt alone does not guarantee factual accuracy.

## Setup on Windows

Use Python 3.11 or 3.12.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set your actual OpenAI API key in .env.

## Run the pipeline

```powershell
python check_dataset.py
python chunk_documents.py
python embed_chunks.py
python vector_store.py
python check_retrieval.py
python check_answers.py
python test_queries.py
python compare_chunking.py
python demo.py
python -m uvicorn api:app --reload
```

Embedding and answer-generation commands use the OpenAI API.
Re-running embed_chunks.py regenerates embeddings and incurs API usage.
Rebuild both embeddings and indexes after changing documents or the embedding model.

Open http://127.0.0.1:8000/docs.

Endpoints:

- GET /health: process health only
- POST /retrieve: retrieved passages and similarity scores
- POST /ask: generated answer and retrieved sources

Example request:

```json
{
  "query": "How long are document backups retained?",
  "top_k": 5,
  "chunk_size": 500
}
```

Blank questions and unsupported chunk sizes are rejected with HTTP 422.
This is a local learning application without authentication.

## Evaluation results

15 questions cover direct facts, concepts, multiple documents and missing answers.
12 questions are answerable; 3 are deliberately unanswerable.

Evidence recall measures the fraction of labeled source excerpts found in
retrieved chunks. Unanswerable questions are excluded from recall.

| Metric | 500 characters | 1000 characters |
|---|---:|---:|
| Mean evidence recall | 1.0000 | 0.9583 |
| Questions with all required evidence | 12/12 | 11/12 |
| Average retrieved characters | 2144 | 3950 |

The 500-character strategy performed better on this small test set and is
the default. It retrieved all labeled evidence with about 46% less text.

These results measure retrieval evidence coverage, not generated-answer accuracy.
Exact excerpt matching can miss alternative useful passages or split evidence.
Fixed top_k also gives larger chunks a larger context budget.

## Observed retrieval failures

For Q10, the 1000-character strategy did not retrieve the required excerpt:
“Audit logs are retained for 180 days.”

Thus, it did not retrieve all evidence needed to answer the combined retention
question. Inspect the saved passages before assigning a specific cause.

The monthly Team price is absent from the dataset. Retrieval still returned
related product information, with top scores of 0.8354 and 0.8110.

Those passages cannot supply the price. The correct behavior is to abstain.
Adding authoritative pricing data could resolve this missing-knowledge case;
increasing top_k cannot recover a fact absent from the dataset.

## Demonstration and reports

demo.py runs eight questions and saves generated answers, citations and
retrieved passages.

Reports:

- reports/retrieval_results.json
- reports/chunking_comparison.txt
- reports/eight_query_demo.json

Review the actual outputs before submission.

