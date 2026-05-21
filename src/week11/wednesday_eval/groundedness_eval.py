"""
Groundedness Evaluator
----------------------
Uses GPT-4o as a judge to score whether each answer
is grounded in the retrieved context.

This is called "LLM-as-judge" — the most powerful evaluation
technique in production RAG systems. You use a strong model
to evaluate the output of your pipeline.

Azure AI Foundry has a built-in groundedness evaluator.
Here we build it from scratch so you understand exactly
what it does inside.

SCORING SCALE:
  5 — Fully grounded. Every claim in the answer is in the context.
  4 — Mostly grounded. Minor phrasing additions but no wrong facts.
  3 — Partially grounded. Some claims in context, some not.
  2 — Mostly ungrounded. Answer uses context loosely, adds inventions.
  1 — Not grounded. Answer contradicts or ignores context entirely.
"""

from openai import AzureOpenAI

AZURE_OAI_ENDPOINT = "https://fayz-openai.openai.azure.com/"
AZURE_OAI_KEY      = "F2FBAVkbe8isc2gqXnSO7HYr4Gh03L8Y5FegiE4DM4yZi9NRfS03JQQJ99CEACYeBjFXJ3w3AAABACOGjfTg"

client = AzureOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version="2024-02-01"
)

GROUNDEDNESS_PROMPT = """You are an expert evaluator measuring groundedness of AI answers.

Groundedness means: does every claim in the ANSWER exist in the CONTEXT?

CONTEXT (retrieved documents):
{context}

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

Score the answer on this scale:
5 = Fully grounded — every single claim in the answer is supported by the context
4 = Mostly grounded — answer is correct but adds minor conversational phrases
3 = Partially grounded — some claims in context, at least one claim not in context
2 = Mostly ungrounded — answer adds significant facts not in context
1 = Not grounded — answer contradicts context or ignores it completely

Special case: if the answer says "I do not have that information" and the context
does NOT contain the answer, score it 5 (correct refusal).
If the answer says "I do not have that information" but the context DOES contain
the answer, score it 1 (wrong refusal — failure to retrieve).

Respond with ONLY a JSON object in this exact format:
{{
  "score": <1-5>,
  "reasoning": "<one sentence explaining the score>",
  "grounded_claims": ["<claim1 from answer that IS in context>"],
  "ungrounded_claims": ["<claim from answer that is NOT in context>"]
}}"""


def evaluate_groundedness(question: str, context: str, answer: str) -> dict:
    """
    Uses GPT-4o to judge if the answer is grounded in the context.
    Returns score 1-5 with reasoning.
    """
    prompt = GROUNDEDNESS_PROMPT.format(
        context=context,
        question=question,
        answer=answer
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON response
    import json
    try:
        # Remove markdown code fences if present
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        result["tokens_used"] = response.usage.total_tokens
        return result
    except Exception:
        return {
            "score": 0,
            "reasoning": f"Parse error: {raw[:100]}",
            "grounded_claims": [],
            "ungrounded_claims": [],
            "tokens_used": response.usage.total_tokens
        }
