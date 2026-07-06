import os
import ssl

import certifi
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone


def setup_ssl():
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    ssl._create_default_https_context = ssl._create_unverified_context


def init_clients():
    """Initialize and return (client, index, embeddings_model, min_relevance_score).

    Expects the Pinecone index to already exist — run rag_pipeline.ipynb first
    to build and populate the index.
    """
    load_dotenv(override=True)
    setup_ssl()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-llm-index")
    min_relevance_score = float(os.getenv("MIN_RELEVANCE_SCORE", "0.75"))

    if not gemini_api_key:
        raise ValueError("Missing GEMINI_API_KEY in environment or .env")
    if not pinecone_api_key:
        raise ValueError("Missing PINECONE_API_KEY in environment or .env")

    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        raise RuntimeError(
            f"האינדקס '{index_name}' לא קיים ב-Pinecone. "
            "הריצו קודם את rag_pipeline.ipynb כדי לבנות ולמלא את האינדקס."
        )
    index = pc.Index(index_name)

    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=gemini_api_key,
    )

    client = genai.Client(api_key=gemini_api_key)

    return client, index, embeddings_model, min_relevance_score


def generate_answer(query, client, index, embeddings_model, min_score, top_k=3):
    """Query Pinecone for relevant chunks and generate an answer with Gemini.

    Returns a dict with keys ``answer`` (str) and ``sources`` (list of dicts
    with ``source``, ``score``, and ``text``).
    """
    query_vector = embeddings_model.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    matches = [m for m in results["matches"] if m["score"] >= min_score]

    if not matches:
        return {
            "answer": "לא נמצא מידע רלוונטי במסמכים.",
            "sources": [],
        }

    context = "\n\n".join(match["metadata"]["text"] for match in matches)

    prompt = f"""Answer the question using only the context below.
If the context does not contain enough information, say you don't know.

Context:
{context}

Question: {query}

Answer:"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
    )

    sources = [
        {
            "source": match["metadata"].get("source", "unknown"),
            "score": match["score"],
            "text": match["metadata"].get("text", ""),
        }
        for match in matches
    ]

    return {"answer": response.text, "sources": sources}
