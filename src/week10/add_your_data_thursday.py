import os
from dotenv import load_dotenv

load_dotenv(), json, time
from openai import AzureOpenAI

AZURE_OAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT_GPT     = "gpt-4o"
DEPLOYMENT_EMBED   = "text-embedding-3-small"
SEARCH_ENDPOINT    = "https://fayz-search.search.windows.net"
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX       = "connectplus-rag"

client = AzureOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version="2024-02-01"
)

os.makedirs("./outputs/week10", exist_ok=True)

def make_data_source(query_type="vector", top_n=5, strictness=4):
    return {
        "type": "azure_search",
        "parameters": {
            "endpoint":   SEARCH_ENDPOINT,
            "index_name": SEARCH_INDEX,
            "authentication": {"type": "api_key", "key": SEARCH_KEY},
            "query_type": query_type,
            "embedding_dependency": {
                "type":            "deployment_name",
                "deployment_name": DEPLOYMENT_EMBED
            },
            "top_n_documents": top_n,
            "strictness":      strictness,
            "fields_mapping": {
                "content_fields": ["content"],
                "title_field":    "title",
                "filepath_field": "source",
                "vector_fields":  ["embedding"]
            }
        }
    }

SYSTEM_MESSAGE = """You are Finn, a ConnectPlus UK customer service assistant.
Answer using ONLY information from the provided source documents.
If the answer is not in the documents say exactly:
I do not have that information. Please contact support@connectplus.co.uk
Use British English. Be concise - maximum 4 sentences."""

def ask(question, query_type="vector", top_n=5, strictness=4, conversation_history=None):
    t0 = time.time()
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=DEPLOYMENT_GPT,
        messages=messages,
        max_tokens=400,
        temperature=0,
        extra_body={"data_sources": [make_data_source(query_type, top_n, strictness)]}
    )
    latency = round((time.time() - t0) * 1000)
    answer  = response.choices[0].message.content

    return {
        "question":   question,
        "answer":     answer,
        "tokens":     response.usage.total_tokens,
        "latency_ms": latency,
        "refused":    "do not have that information" in answer.lower()
    }

print("=" * 60)
print("ADD YOUR DATA — Thursday Week 10")
print("=" * 60)

# TEST 1 — Standard queries
print("\n── TEST 1: Standard queries ──\n")
queries = [
    "What is the cancellation fee if I cancel after 6 months?",
    "What is the maximum discount an agent can offer?",
    "What happens if my direct debit fails three times?",
    "Does ConnectPlus offer 5G mobile plans?",
]
results = []
for q in queries:
    r = ask(q)
    status = "REFUSED" if r["refused"] else "ANSWERED"
    print(f"Q: {q}")
    print(f"  [{status}] {r['tokens']} tokens | {r['latency_ms']}ms")
    print(f"  A: {r['answer'][:200]}\n")
    results.append(r)

# TEST 2 — Strictness comparison
print("── TEST 2: Strictness levels ──\n")
for strictness in [1, 3, 5]:
    r = ask("Does ConnectPlus offer 5G mobile plans?", strictness=strictness)
    status = "REFUSED" if r["refused"] else "ANSWERED"
    print(f"  Strictness {strictness}: [{status}] {r['answer'][:100]}")

# TEST 3 — Multi-turn conversation
print("\n── TEST 3: Multi-turn conversation ──\n")
history = []
turns = [
    "I want to cancel my ConnectPlus contract.",
    "I have been a customer for 18 months.",
    "I was also charged twice last month. What about my refund?",
]
for turn in turns:
    r = ask(turn, conversation_history=history)
    print(f"  User: {turn}")
    print(f"  Finn: {r['answer'][:200]}\n")
    history.append({"role": "user",      "content": turn})
    history.append({"role": "assistant", "content": r["answer"]})

# Save results
with open("./outputs/week10/thursday_add_your_data.json", "w") as f:
    json.dump(results, f, indent=2)

print("=" * 60)
print("Thursday Week 10 COMPLETE")
print("Saved: outputs/week10/thursday_add_your_data.json")
print("=" * 60)
