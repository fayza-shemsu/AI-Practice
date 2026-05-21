import os, json, math, time
from openai import AzureOpenAI

AZURE_OAI_ENDPOINT = "https://fayz-openai.openai.azure.com/"
AZURE_OAI_KEY      = "F2FBAVkbe8isc2gqXnSO7HYr4Gh03L8Y5FegiE4DM4yZi9NRfS03JQQJ99CEACYeBjFXJ3w3AAABACOGjfTg"
DEPLOYMENT_EMBED   = "text-embedding-3-small"

client = AzureOpenAI(
    azure_endpoint = AZURE_OAI_ENDPOINT,
    api_key        = AZURE_OAI_KEY,
    api_version    = "2024-02-01"
)

os.makedirs("./outputs/week10", exist_ok=True)

def embed_batch(texts):
    if isinstance(texts, str):
        texts = [texts]
    r = client.embeddings.create(model=DEPLOYMENT_EMBED, input=texts)
    return [d.embedding for d in r.data]

def cosine_sim(a, b):
    dot   = sum(x*y for x,y in zip(a,b))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0

def bm25_score(query_terms, doc_text, k1=1.5, b=0.75):
    doc_terms = doc_text.lower().split()
    doc_len   = len(doc_terms)
    avg_len   = 60
    score     = 0.0
    for term in query_terms:
        tf = doc_terms.count(term.lower())
        if tf == 0:
            continue
        idf     = math.log((len(doc_terms) + 1) / (tf + 0.5))
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
        score  += idf * tf_norm
    return score

def rrf_fusion(ranking_a, ranking_b, k=60):
    scores = {}
    for rank, doc_id in enumerate(ranking_a):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
    for rank, doc_id in enumerate(ranking_b):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

DOCUMENTS = [
    {"id": "cancel_p1", "source": "cancellation_policy.pdf", "page": 1,
     "content": "ConnectPlus cancellation requires 30 days written notice. No fee within first 14 days cooling off period. After 14 days 25 pound early termination fee applies for contracts under 12 months. After 12 months no termination fee. Equipment must be returned within 14 days or 75 pound non-return fee applies."},

    {"id": "cancel_p2", "source": "cancellation_policy.pdf", "page": 2,
     "content": "Refunds for overpayments and double billing charged twice processed within 5 to 7 working days to original payment method. Direct debit refunds take up to 10 working days. Disputed charges must be raised within 60 days."},

    {"id": "retention_p3", "source": "retention_playbook.pdf", "page": 3,
     "content": "HIGH risk customers 3 or more support calls or explicit cancel intent: offer 3 month loyalty discount maximum 20 percent off monthly bill and assign dedicated Tier 2 agent priority engineer visit 48 hours. MEDIUM risk billing dispute: one month bill credit and free router upgrade. LOW risk: satisfaction survey and 20 pound referral bonus."},

    {"id": "products_p7", "source": "product_catalogue.pdf", "page": 7,
     "content": "ConnectPlus broadband plans. Essential 35Mbps 25 pounds per month no minimum term. Standard 67Mbps 35 pounds per month 12 month contract. Premium 150Mbps 50 pounds per month 12 month contract. Ultrafast 500Mbps 70 pounds per month 24 month contract. Speed Promise exit without penalty if below 50 percent advertised speed for 3 consecutive days."},

    {"id": "billing_p2", "source": "billing_policy.pdf", "page": 2,
     "content": "Bills generated 1st of each month. Payment due within 14 days. Late payment fee 12 pounds after 14 day grace period. Double billing charged twice in error refund within 3 to 5 working days. Direct debit failure 5 pound admin fee three consecutive failures account suspension."},

    {"id": "technical_p5", "source": "technical_support.pdf", "page": 5,
     "content": "Internet dropping at specific times such as 9pm usually peak time congestion. Engineer visit to check line quality. Router restart hold reset button 10 seconds wait 2 minutes. Speed below 50 percent of plan log 3 consecutive days then call 0800 123 456 to invoke Speed Promise."},

    {"id": "billing_p3", "source": "billing_policy.pdf", "page": 3,
     "content": "ConnectPlus must provide minimum 30 days written notice before any price increase. Customers may exit contract without penalty if they do not accept a price increase and notify ConnectPlus within 30 days of the notice."},
]

print("=" * 60)
print("VECTOR SEARCH — Tuesday Week 10")
print("=" * 60)

print("\n── STEP 1: Indexing ──")
start      = time.time()
contents   = [d["content"] for d in DOCUMENTS]
embeddings = embed_batch(contents)
for doc, emb in zip(DOCUMENTS, embeddings):
    doc["embedding"] = emb
elapsed = time.time() - start
print(f"  Documents indexed : {len(DOCUMENTS)}")
print(f"  Time              : {elapsed:.2f}s")
print(f"  Memory (float32)  : {len(DOCUMENTS) * 1536 * 4 / 1024:.1f} KB")
print(f"  At 1M docs        : {1_000_000 * 1536 * 4 / 1024**3:.1f} GB — need real vector DB")

QUERIES = [
    ("How do I cancel and what is the fee?",           "cancel_p1"),
    ("I was charged twice this month",                 "billing_p2"),
    ("internet drops every evening at 9pm",            "technical_p5"),
    ("best offer for customer threatening to leave",   "retention_p3"),
    ("can the company raise my price without warning", "billing_p3"),
]

print("\n── STEP 2: Pure vector search ──")
print(f"  {'Query':<46} {'Got':<15} {'Score':<7} {'OK?'}")
print(f"  {'-'*72}")
vector_rankings = {}
for query, expected in QUERIES:
    q_emb  = embed_batch([query])[0]
    scored = sorted([(cosine_sim(q_emb, d["embedding"]), d) for d in DOCUMENTS], reverse=True)
    top    = scored[0][1]
    score  = scored[0][0]
    ok     = "YES" if top["id"] == expected else f"NO  (want {expected})"
    print(f"  {query[:45]:<46} {top['id']:<15} {score:.4f}  {ok}")
    vector_rankings[query] = [s[1]["id"] for s in scored]

print("\n── STEP 3: BM25 keyword search ──")
print(f"  {'Query':<46} {'Got':<15} {'Score':<7} {'OK?'}")
print(f"  {'-'*72}")
keyword_rankings = {}
for query, expected in QUERIES:
    terms  = query.lower().split()
    scored = sorted([(bm25_score(terms, d["content"]), i, d) for i, d in enumerate(DOCUMENTS)], reverse=True)
    scored = [(s, d) for s, i, d in scored]
    top    = scored[0][1]
    score  = scored[0][0]
    ok     = "YES" if top["id"] == expected else f"NO  (want {expected})"
    print(f"  {query[:45]:<46} {top['id']:<15} {score:.4f}  {ok}")
    keyword_rankings[query] = [s[1]["id"] for s in scored]

print("\n── STEP 4: Hybrid search (RRF fusion) ──")
print(f"  {'Query':<46} {'Got':<15} {'OK?'}")
print(f"  {'-'*68}")
correct = 0
for query, expected in QUERIES:
    fused  = rrf_fusion(vector_rankings[query], keyword_rankings[query])
    top_id = fused[0]
    ok     = "YES" if top_id == expected else f"NO  (want {expected})"
    if top_id == expected:
        correct += 1
    print(f"  {query[:45]:<46} {top_id:<15} {ok}")
print(f"\n  Hybrid accuracy: {correct}/{len(QUERIES)}")

print("\n── STEP 5: Score threshold — when to refuse ──")
THRESHOLD = 0.35
test_queries = [
    "what is the weather in London",
    "how do I cook pasta",
    "what is the cancellation fee",
]
for query in test_queries:
    q_emb  = embed_batch([query])[0]
    scored = sorted([(cosine_sim(q_emb, d["embedding"]), d) for d in DOCUMENTS], reverse=True)
    score  = scored[0][0]
    action = "ANSWER" if score >= THRESHOLD else "REFUSE — not in knowledge base"
    print(f"  [{score:.4f}] {action}")
    print(f"           Q: '{query}'")

print("\n── STEP 6: Chunking strategy ──")
text = "ConnectPlus cancellation requires 30 days notice. No fee within 14 days cooling off period. After 14 days 25 pound fee applies. Equipment must be returned within 14 days or 75 pound charge. Refunds processed within 5 to 7 working days."
for size, overlap in [(10, 0), (10, 3), (20, 5)]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    print(f"  size={size} overlap={overlap} → {len(chunks)} chunks")
    print(f"  chunk 1: '{chunks[0]}'")
    print(f"  chunk 2: '{chunks[1] if len(chunks)>1 else 'N/A'}'")
    print()

with open("./outputs/week10/tuesday_results.json", "w") as f:
    json.dump({"documents": len(DOCUMENTS), "hybrid_accuracy": f"{correct}/{len(QUERIES)}"}, f, indent=2)

print("=" * 60)
print("Tuesday Week 10 COMPLETE")
print("=" * 60)