"""
Test the deployed RAG endpoint.
Same pattern as Week 6 curl test — but in Python.
"""
import json
import urllib.request
import urllib.error

# Fill these in after deployment completes
SCORING_URI = "YOUR_SCORING_URI_HERE"
API_KEY     = "YOUR_API_KEY_HERE"

def call_endpoint(question: str) -> dict:
    """Call the live RAG endpoint — same as curl from Week 6."""
    payload = json.dumps({"question": question}).encode("utf-8")

    req = urllib.request.Request(
        SCORING_URI,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
    )

    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())

# Test questions
questions = [
    "What is the cancellation fee after 6 months?",
    "What is the maximum discount an agent can offer?",
    "Does ConnectPlus offer 5G roaming in Japan?",
]

print("Testing live RAG endpoint...")
print("=" * 60)

for q in questions:
    print(f"\nQ: {q}")
    try:
        result = call_endpoint(q)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Scores:  {result['retrieval_scores']}")
        print(f"Tokens:  {result['tokens_used']}")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 60)
print("Endpoint test complete")
