"""
Week 11 Thursday — RAG Flow Scoring Script
This is the same pattern as Week 6 score.py but instead of
a churn model, the service runs the full RAG pipeline.

Week 6:  init() loaded a .pkl model  → run() predicted churn
Week 11: init() loads connections    → run() runs RAG pipeline

The deployment concept is identical. The payload changed.
"""
import os
from dotenv import load_dotenv

load_dotenv()
import json
import logging
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)

# Global clients — loaded once in init(), reused for every request
oai_client    = None
search_client = None

AZURE_OAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OAI_KEY = os.getenv("AZURE_OPENAI_KEY")
SEARCH_ENDPOINT    = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX       = "connectplus-rag"


def init():
    """
    Called ONCE when the endpoint starts up.
    Loads all clients into memory so every request is fast.

    Week 6 lesson: init() loads the model once.
    Here:          init() creates the API clients once.
    Same concept — expensive setup done once, not per request.
    """
    global oai_client, search_client

    logger.info("Loading RAG pipeline clients...")

    oai_client = AzureOpenAI(
        azure_endpoint=AZURE_OAI_ENDPOINT,
        api_key=AZURE_OAI_KEY,
        api_version="2024-02-01"
    )

    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_KEY)
    )

    logger.info("RAG pipeline ready")


def run(raw_data):
    """
    Called for EVERY request.
    Input:  JSON string with 'question' field
    Output: JSON string with 'answer', 'sources', 'retrieval_scores'

    Same pattern as Week 6 — accept JSON, return JSON.
    """
    try:
        data     = json.loads(raw_data)
        question = data.get("question", "")

        if not question:
            return json.dumps({"error": "No question provided"})

        # Step 1 — Embed the question
        vec_response = oai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[question]
        )
        question_vector = vec_response.data[0].embedding

        # Step 2 — Retrieve relevant chunks
        search_results = search_client.search(
            search_text=question,
            vector_queries=[VectorizedQuery(
                vector=question_vector,
                k_nearest_neighbors=3,
                fields="embedding"
            )],
            select=["content", "source", "title"],
            top=3
        )

        chunks  = []
        sources = []
        scores  = []

        for r in search_results:
            chunks.append(r["content"])
            sources.append(r["source"])
            scores.append(round(r["@search.score"], 4))

        context = "\n\n---\n\n".join(chunks)

        # Step 3 — Generate grounded answer
        prompt = f"""You are Finn, a ConnectPlus UK customer service assistant.
Answer using ONLY information from the context below.
If the answer is not in the context say: I do not have that information.
Use British English. Maximum 3 sentences.

Context:
{context}

Question: {question}"""

        llm_response = oai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0
        )

        answer = llm_response.choices[0].message.content
        tokens = llm_response.usage.total_tokens
        finish = llm_response.choices[0].finish_reason

        return json.dumps({
            "answer":           answer,
            "sources":          list(set(sources)),
            "retrieval_scores": scores,
            "tokens_used":      tokens,
            "finish_reason":    finish
        })

    except Exception as e:
        logger.error(f"Error in run(): {e}")
        return json.dumps({"error": str(e)})
