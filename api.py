from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from openai import OpenAIError

from retrieval import retrieve
from rag import answer_question

# Create the API application.
app = FastAPI(title="Complete Text RAG Pipeline")


# Describe and validate the JSON accepted by our endpoints.
class QueryRequest(BaseModel):

    query: str = Field(min_length=1, max_length=4000)

    top_k: int = Field(default=5, ge=1, le=20)

    chunk_size: Literal[500, 1000] = 500

    # Reject questions containing only spaces.
    @field_validator("query")
    @classmethod
    def validate_query(cls, value):

        value = value.strip()

        if not value:
            raise ValueError("The query cannot be blank.")

        return value


@app.get("/health")
def health():

    return {"status": "ok"}


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):

    try:
        results = retrieve(
            query=request.query,
            top_k=request.top_k,
            chunk_size=request.chunk_size
        )

        return {
            "query": request.query,
            "chunk_size": request.chunk_size,
            "results": results
        }

    except OpenAIError as error:
        raise HTTPException(
            status_code=502,
            detail="OpenAI request failed. Check API access and quota."
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error


@app.post("/ask")
def ask_endpoint(request: QueryRequest):

    try:
        result = answer_question(
            query=request.query,
            top_k=request.top_k,
            chunk_size=request.chunk_size
        )

        return result

    except OpenAIError as error:
        raise HTTPException(
            status_code=502,
            detail="OpenAI request failed. Check API access and quota."
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error