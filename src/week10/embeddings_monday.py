import os
import json
import math
import time
from openai import AzureOpenAI

AZURE_OAI_ENDPOINT = "https://fayz-openai.openai.azure.com/"
AZURE_OAI_KEY      = "F2FBAVkbe8isc2gqXnSO7HYr4Gh03L8Y5FegiE4DM4yZi9NRfS03JQQJ99CEACYeBjFXJ3w3AAABACOGjfTg"
DEPLOYMENT_EMBED   = "text-embedding-3-small"

client = AzureOpenAI(
    azure_endpoint =  "https://fayz-openai.openai.azure.com/",
    api_key        = "F2FBAVkbe8isc2gqXnSO7HYr4Gh03L8Y5FegiE4DM4yZi9NRfS03JQQJ99CEACYeBjFXJ3w3AAABACOGjfTg",
    api_version    = "2024-02-01"
)

os.makedirs("./outputs/week10", exist_ok=True)

# ── Core functions ────────────────────────────────────────────
def embed(texts):
    """Embed one string or list of strings. Always returns list of embeddings."""
    if isinstance(texts, str):
        texts = [texts]
    # Strip empty strings — API rejects them
    texts = [t.strip() for t in texts if t.strip()]
    r = client.embeddings.create(
        model = DEPLOYMENT_EMBED,
        input = texts
    )
    return [d.embedding for d in r.data], r.usage.total_tokens

def normalise(v):
    mag = math.sqrt(sum(x**2 for x in v))
    return [x/mag for x in v] if mag > 0 else v

def cosine_sim(a, b):
    return sum(x*y for x,y in zip(a,b)) / (
        math.sqrt(sum(x**2 for x in a)) *
        math.sqrt(sum(x**2 for x in b))
    )

def interpret_score(score):
    if score > 0.55: return "VERY SIMILAR"
    if score > 0.40: return "RELATED"
    if score > 0.25: return "LOOSELY RELATED"
    return "DIFFERENT TOPIC"

print("=" * 60)
print("EMBEDDINGS — Monday Week 10")
print("=" * 60)
results = {}

# ── TEST 1: Basic embedding — what does a vector look like? ───
print("\n── TEST 1: Basic embedding structure ──")
embeddings, tokens = embed("Customer churn in telecoms")
vec = embeddings[0]
print(f"  Input:      'Customer churn in telecoms'")
print(f"  Dimensions: {len(vec)}")
print(f"  Tokens used:{tokens}")
print(f"  First 8 values: {[round(x,4) for x in vec[:8]]}")
print(f"  Last 8 values:  {[round(x,4) for x in vec[-8:]]}")
print(f"  Min value:  {min(vec):.4f}")
print(f"  Max value:  {max(vec):.4f}")
print(f"  Magnitude:  {math.sqrt(sum(x**2 for x in vec)):.4f}")
results["test1_dims"] = len(vec)

# ── TEST 2: Semantic similarity — meaning vs keywords ─────────
print("\n── TEST 2: Semantic similarity ──")
pairs = [
    ("cancel my subscription",     "I want to stop my service"),
    ("cancel my subscription",     "how do I leave ConnectPlus"),
    ("cancel my subscription",     "my bill is too high"),
    ("cancel my subscription",     "what is the weather today"),
    ("I am happy with the service","great experience overall"),
    ("I am happy with the service","terrible, worst company ever"),
]

embeddings_list = []
all_texts = [t for pair in pairs for t in pair]
all_embeddings, tokens = embed(all_texts)
print(f"  Embedded {len(all_texts)} texts in one API call. Tokens: {tokens}")
print()

pair_results = []
for i, (a, b) in enumerate(pairs):
    ea = all_embeddings[i*2]
    eb = all_embeddings[i*2+1]
    score = cosine_sim(ea, eb)
    interp = interpret_score(score)
    print(f"  [{score:.4f}] {interp}")
    print(f"    A: '{a}'")
    print(f"    B: '{b}'")
    print()
    pair_results.append({"a":a,"b":b,"score":round(score,4),"interpretation":interp})

results["semantic_pairs"] = pair_results

# ── TEST 3: Semantic search — find most relevant document ─────
print("── TEST 3: Semantic search ──")
documents = [
    "ConnectPlus cancellation requires 30 days notice. £25 fee after 14 days.",
    "Monthly bills generated on 1st. Late payment fee £12 after 14 days.",
    "High risk customers: offer 20% discount for 3 months plus dedicated agent.",
    "Broadband plans: Essential 35Mbps £25/month, Premium 150Mbps £50/month.",
    "Internet dropping at 9pm is likely peak-time congestion. Engineer visit resolves.",
    "Refunds for overpayments processed within 5-7 working days.",
    "ConnectPlus offers 24/7 UK-based customer support on all plans.",
]

# Embed all documents in one call
doc_embeddings, doc_tokens = embed(documents)
print(f"  Embedded {len(documents)} documents. Tokens: {doc_tokens}")

queries = [
    "How do I cancel? Is there a fee?",
    "My internet is slow every evening",
    "What can I offer an angry customer threatening to leave?",
    "I was charged twice this month",
]

search_results = {}
for query in queries:
    q_emb, q_tok = embed(query)
    q_vec = q_emb[0]
    scored = sorted(
        [(cosine_sim(q_vec, d_emb), doc)
         for d_emb, doc in zip(doc_embeddings, documents)],
        reverse=True
    )
    print(f"\n  Query: '{query}'")
    for score, doc in scored[:3]:
        marker = " ← TOP" if score == scored[0][0] else ""
        print(f"    [{score:.4f}] {doc[:70]}{marker}")
    search_results[query] = {"top_result": scored[0][1], "score": round(scored[0][0],4)}

results["semantic_search"] = search_results

# ── TEST 4: Clustering — group similar texts ──────────────────
print("\n── TEST 4: Clustering by meaning ──")
mixed_texts = [
    "I want to cancel my account",
    "How do I stop my subscription",
    "I am leaving ConnectPlus",
    "My bill is wrong this month",
    "I was charged the wrong amount",
    "There is an error on my invoice",
    "My internet keeps dropping",
    "The connection is very slow",
    "Wi-Fi cuts out every evening",
]

mixed_embeddings, _ = embed(mixed_texts)

# Simple clustering: group by highest mutual similarity
def simple_cluster(texts, embeddings, threshold=0.80):
    clusters = []
    assigned = set()
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        if i in assigned:
            continue
        cluster = [text]
        assigned.add(i)
        for j, (other_text, other_emb) in enumerate(zip(texts, embeddings)):
            if j in assigned:
                continue
            if cosine_sim(emb, other_emb) >= threshold:
                cluster.append(other_text)
                assigned.add(j)
        clusters.append(cluster)
    return clusters

clusters = simple_cluster(mixed_texts, mixed_embeddings, threshold=0.50)
print(f"  {len(mixed_texts)} texts → {len(clusters)} clusters (threshold=0.50)\n")
for i, cluster in enumerate(clusters):
    print(f"  Cluster {i+1} ({len(cluster)} items):")
    for item in cluster:
        print(f"    - {item}")
results["clusters"] = clusters

# ── TEST 5: Dimensions matter — 1536 vs reduced ───────────────
print("\n── TEST 5: Dimension reduction effect ──")
try:
    full_emb,    _ = embed("cancel my subscription")
    r_small = client.embeddings.create(
        model=DEPLOYMENT_EMBED,
        input=["cancel my subscription", "I want to stop my service"],
        dimensions=256
    )
    reduced_a = r_small.data[0].embedding
    reduced_b = r_small.data[1].embedding

    r_full = client.embeddings.create(
        model=DEPLOYMENT_EMBED,
        input=["cancel my subscription", "I want to stop my service"]
    )
    full_a = r_full.data[0].embedding
    full_b = r_full.data[1].embedding

    score_full    = cosine_sim(full_a, full_b)
    score_reduced = cosine_sim(reduced_a, reduced_b)

    print(f"  Full 1536 dims:    similarity = {score_full:.4f} — {len(full_a)} numbers stored per text")
    print(f"  Reduced 256 dims:  similarity = {score_reduced:.4f} — {len(reduced_a)} numbers stored per text")
    print(f"  Storage reduction: {len(full_a)/len(reduced_a):.0f}× smaller")
    print(f"  Quality change:    {abs(score_full-score_reduced):.4f} difference")
    results["dimension_reduction"] = {
        "full_1536": round(score_full,4),
        "reduced_256": round(score_reduced,4)
    }
except Exception as e:
    print(f"  Dimension reduction: {e}")

# ── TEST 6: Token cost awareness ─────────────────────────────
print("\n── TEST 6: Embedding cost analysis ──")
short_text  = "cancel"
medium_text = "I want to cancel my ConnectPlus subscription due to poor service"
long_text   = " ".join(["Customer churn analysis report"] * 50)

for label, text in [("Short (1 word)", short_text),
                     ("Medium (12 words)", medium_text),
                     ("Long (150 words)", long_text)]:
    _, tok = embed(text)
    cost = tok * 0.02 / 1_000_000
    print(f"  {label:<20} {tok:>5} tokens   ${cost:.8f} per embedding")
    print(f"    At 1M embeddings/day: ${cost*1_000_000:.2f}/day")

print()
print("  Embedding is cheap — 1M embeddings of medium text ≈ $0.02/day")
print("  The expensive part is GPT-4o generation, not embedding")

with open("./outputs/week10/monday_embeddings.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("Monday Week 10 COMPLETE")
