import os, json, time, re
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType,
    SimpleField, SearchableField, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields,
    SemanticField
)
from azure.core.credentials import AzureKeyCredential

# ── Credentials ───────────────────────────────────────────────
AZURE_OAI_ENDPOINT = "https://fayz-openai.openai.azure.com/"
AZURE_OAI_KEY      = "F2FBAVkbe8isc2gqXnSO7HYr4Gh03L8Y5FegiE4DM4yZi9NRfS03JQQJ99CEACYeBjFXJ3w3AAABACOGjfTg"
SEARCH_ENDPOINT    = "https://fayz-search.search.windows.net"
SEARCH_KEY         = "P8vmYuqS7rOctpch0i8SMVFjOBokUtCpufq9B1s9cmAzSeCFyHJC"
DEPLOY_EMBED       = "text-embedding-3-small"
DEPLOY_GPT         = "gpt-4o"

oai_client = AzureOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version="2024-02-01"
)

os.makedirs("./outputs/week10", exist_ok=True)

# ── Full source documents (raw text) ─────────────────────────
# In production these would be loaded from files.
# Here we define them inline so the experiment is self-contained.
RAW_DOCUMENTS = [
    {
        "source": "cancellation_policy.txt",
        "title": "Cancellation Policy",
        "text": """ConnectPlus cancellation requires 30 days written notice to cancellations@connectplus.co.uk. No fee within first 14 days cooling-off period under Consumer Rights Act 2015. After 14 days there is a £25 early termination fee for contracts under 12 months. After 12 months there is no termination fee. After 24 months the customer is free to leave anytime without any penalty. Equipment return is required within 14 days of cancellation. Non-return fee is £75 added to the final bill. Return label is sent via email within 2 working days of cancellation request. Final bill is issued within 5 working days. Pro-rata refund is calculated from the cancellation date. Refunds are sent to the original payment method within 7 working days."""
    },
    {
        "source": "retention_playbook.txt",
        "title": "Retention Playbook",
        "text": """HIGH RISK is defined as two or more of the following signals: explicit cancel intent, 3 or more support calls in 30 days, plan downgrade in last 14 days, billing complaint this month. HIGH RISK actions require acknowledging frustration first, offering 20% loyalty discount for 3 months maximum, assigning a dedicated Tier-2 agent, and arranging a priority engineer visit within 48 hours if there is a technical issue. MEDIUM RISK means one signal is present. Actions include one-month bill credit applied immediately, free router upgrade if the router is over 2 years old, and a satisfaction survey at the end of the call. LOW RISK means general dissatisfaction with no specific signals. Actions include an NPS survey, £20 referral bonus credit, and information about the loyalty rewards programme. Escalate to supervisor immediately if the customer mentions legal action, Ofcom, media threat, safeguarding concern, or bereavement. Discount authorisation limits are as follows: Agent maximum is 20% for 3 months. Team Leader maximum is 30% for 6 months. Manager maximum is 50% for 12 months and requires written approval."""
    },
    {
        "source": "billing_policy.txt",
        "title": "Billing Policy",
        "text": """Bills are generated on the 1st of each month for the following month. Payment is due within 14 days. Late payment fee is £12 after the grace period expires. Double billing errors are refunded within 3 to 5 working days. Overcharge disputes must be raised within 60 days via billing@connectplus.co.uk. Direct debit failure on the first attempt incurs a £5 admin fee and payment is retried after 5 working days. Second failure incurs an additional £5 fee. Third failure results in account suspension and a reconnection fee applies when the account is reactivated. Price increases require a minimum of 30 days written notice as required by law. The customer may exit the contract without penalty within 30 days of receiving the price increase notice."""
    },
]

# ── Chunking functions ────────────────────────────────────────

def chunk_fixed_size(text, chunk_size, overlap):
    """
    Strategy 1: Fixed size chunking.
    Split text every chunk_size characters with overlap.
    Simplest approach — used for comparison baseline.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]  # drop tiny tail chunks

def chunk_sentence_aware(text, chunk_size, overlap):
    """
    Strategy 2: Sentence-aware chunking.
    Never cuts a sentence in half. Splits at sentence boundaries
    closest to the target chunk_size. Better quality than fixed size.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += " " + sentence
        else:
            if current:
                chunks.append(current.strip())
            # Start new chunk with overlap from previous
            if chunks and overlap > 0:
                # Carry last N chars of previous chunk as overlap
                prev = chunks[-1]
                current = prev[-overlap:] + " " + sentence
            else:
                current = sentence
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]

# ── Index management ──────────────────────────────────────────

def create_index(index_name):
    """Create a fresh search index for this chunk size experiment."""
    idx_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))

    # Delete if exists (clean experiment)
    try:
        idx_client.delete_index(index_name)
        time.sleep(2)
    except:
        pass

    fields = [
        SimpleField(name="id",      type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source",  type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="title",   type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )
    idx_client.create_index(index)
    print(f"  Created index: {index_name}")

def upload_chunks(index_name, chunks_with_meta):
    """Embed and upload chunks to the search index."""
    search_client = SearchClient(
        SEARCH_ENDPOINT, index_name, AzureKeyCredential(SEARCH_KEY)
    )

    # Embed all chunks in one batch
    texts = [c["content"] for c in chunks_with_meta]
    response = oai_client.embeddings.create(model=DEPLOY_EMBED, input=texts)
    embeddings = [d.embedding for d in response.data]

    docs = []
    for i, (chunk, embedding) in enumerate(zip(chunks_with_meta, embeddings)):
        docs.append({
            "id":          f"chunk_{i}",
            "content":     chunk["content"],
            "source":      chunk["source"],
            "title":       chunk["title"],
            "chunk_index": i,
            "embedding":   embedding
        })

    result = search_client.upload_documents(documents=docs)
    succeeded = sum(1 for r in result if r.succeeded)
    return succeeded

def vector_search(index_name, query, top_k=3):
    """Search the index using vector similarity."""
    search_client = SearchClient(
        SEARCH_ENDPOINT, index_name, AzureKeyCredential(SEARCH_KEY)
    )

    # Embed the query
    q_response = oai_client.embeddings.create(model=DEPLOY_EMBED, input=[query])
    query_vector = q_response.data[0].embedding

    from azure.search.documents.models import VectorizedQuery
    results = search_client.search(
        search_text=None,
        vector_queries=[VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )],
        select=["content", "source", "chunk_index"],
        top=top_k
    )
    return [{"content": r["content"], "source": r["source"]} for r in results]

def ask_with_context(question, context_chunks):
    """Send question + retrieved chunks to GPT-4o and get answer."""
    context = "\n\n---\n\n".join([c["content"] for c in context_chunks])

    system = """You are Finn, a ConnectPlus customer service assistant.
Answer using ONLY the provided context below.
If the answer is not in the context, say: I do not have that information.
Be concise — maximum 3 sentences."""

    t0 = time.time()
    response = oai_client.chat.completions.create(
        model=DEPLOY_GPT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=300,
        temperature=0
    )
    latency = round((time.time() - t0) * 1000)
    answer = response.choices[0].message.content
    tokens = response.usage.total_tokens
    return answer, tokens, latency

# ── Evaluation questions ──────────────────────────────────────
# These are designed to test different retrieval challenges:
# Q1 — Simple fact (small chunk should win)
# Q2 — Multi-step reasoning (needs context around the fact)
# Q3 — Complex multi-document (needs content from 2 sources)
# Q4 — Out of scope (both should refuse)

EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What is the cancellation fee after 6 months?",
        "expected_keyword": "£25",
        "type": "simple_fact"
    },
    {
        "id": "Q2",
        "question": "I have been a customer for 18 months. Can I cancel without any fee, and how long do I have to return the equipment?",
        "expected_keyword": "14 days",
        "type": "multi_sentence"
    },
    {
        "id": "Q3",
        "question": "I want to cancel and I also had a billing error last month. What are the refund timelines for both?",
        "expected_keyword": "working days",
        "type": "multi_document"
    },
    {
        "id": "Q4",
        "question": "Does ConnectPlus offer 5G mobile plans?",
        "expected_keyword": None,
        "type": "out_of_scope"
    },
]

def score_answer(answer, expected_keyword, question_type):
    """
    Simple scoring function.
    In production you would use an LLM-as-judge or human raters.
    Here we use keyword presence as a proxy for correctness.
    """
    answer_lower = answer.lower()
    refused = "do not have that information" in answer_lower

    if question_type == "out_of_scope":
        return 1.0 if refused else 0.0  # Should refuse

    if expected_keyword and expected_keyword.lower() in answer_lower:
        return 1.0  # Correct answer
    elif refused:
        return 0.0  # Wrong — should have answered
    else:
        return 0.5  # Answered but keyword missing — partial credit

# ── Main experiment loop ──────────────────────────────────────

CHUNK_CONFIGS = [
    {"chunk_size": 200,  "overlap": 20,  "strategy": "fixed"},
    {"chunk_size": 500,  "overlap": 50,  "strategy": "fixed"},
    {"chunk_size": 1000, "overlap": 100, "strategy": "fixed"},
    {"chunk_size": 2000, "overlap": 200, "strategy": "fixed"},
    {"chunk_size": 500,  "overlap": 50,  "strategy": "sentence"},
    {"chunk_size": 1000, "overlap": 100, "strategy": "sentence"},
]

print("=" * 70)
print("FRIDAY WEEK 10 — Chunk Size Optimization Experiment")
print("=" * 70)
print(f"\nTesting {len(CHUNK_CONFIGS)} configurations × {len(EVAL_QUESTIONS)} questions")
print("This will take a few minutes...\n")

all_results = []

for config in CHUNK_CONFIGS:
    chunk_size = config["chunk_size"]
    overlap    = config["overlap"]
    strategy   = config["strategy"]
    index_name = f"chunk-exp-{strategy}-{chunk_size}"

    print(f"\n{'='*70}")
    print(f"CONFIG: strategy={strategy} | chunk_size={chunk_size} | overlap={overlap}")
    print(f"{'='*70}")

    # Step 1 — Chunk all documents
    all_chunks = []
    for doc in RAW_DOCUMENTS:
        if strategy == "fixed":
            chunks = chunk_fixed_size(doc["text"], chunk_size, overlap)
        else:
            chunks = chunk_sentence_aware(doc["text"], chunk_size, overlap)

        for chunk in chunks:
            all_chunks.append({
                "content": chunk,
                "source":  doc["source"],
                "title":   doc["title"]
            })

    print(f"  Chunks created: {len(all_chunks)}")
    avg_len = sum(len(c["content"]) for c in all_chunks) / len(all_chunks)
    print(f"  Average chunk length: {avg_len:.0f} chars")

    # Step 2 — Create index and upload
    create_index(index_name)
    time.sleep(3)  # Let index propagate
    uploaded = upload_chunks(index_name, all_chunks)
    print(f"  Uploaded: {uploaded}/{len(all_chunks)} chunks")
    time.sleep(2)  # Let index settle before querying

    # Step 3 — Run eval questions
    config_results = []
    total_score  = 0
    total_tokens = 0
    total_latency = 0

    for eq in EVAL_QUESTIONS:
        # Retrieve
        retrieved = vector_search(index_name, eq["question"], top_k=3)

        # Generate
        answer, tokens, latency = ask_with_context(eq["question"], retrieved)

        # Score
        score = score_answer(answer, eq["expected_keyword"], eq["type"])
        total_score   += score
        total_tokens  += tokens
        total_latency += latency

        result = {
            "question_id":   eq["id"],
            "question_type": eq["type"],
            "answer":        answer,
            "score":         score,
            "tokens":        tokens,
            "latency_ms":    latency,
            "chunks_retrieved": len(retrieved),
            "first_chunk_preview": retrieved[0]["content"][:80] if retrieved else ""
        }
        config_results.append(result)

        status = "✅" if score == 1.0 else ("⚠️" if score == 0.5 else "❌")
        print(f"\n  {status} [{eq['id']}] {eq['type']}")
        print(f"     Q: {eq['question'][:60]}")
        print(f"     A: {answer[:120]}")
        print(f"     Score: {score} | Tokens: {tokens} | Latency: {latency}ms")

    avg_score = total_score / len(EVAL_QUESTIONS)
    print(f"\n  ── Config Summary ──")
    print(f"  Avg Score:   {avg_score:.2f} / 1.00")
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Avg Latency:  {total_latency // len(EVAL_QUESTIONS)}ms")

    all_results.append({
        "config":        config,
        "index_name":    index_name,
        "num_chunks":    len(all_chunks),
        "avg_chunk_len": round(avg_len),
        "avg_score":     round(avg_score, 3),
        "total_tokens":  total_tokens,
        "avg_latency_ms": total_latency // len(EVAL_QUESTIONS),
        "question_results": config_results
    })

# ── Final comparison table ────────────────────────────────────
print("\n\n" + "=" * 70)
print("FINAL RESULTS — All Configurations Ranked by Score")
print("=" * 70)

sorted_results = sorted(all_results, key=lambda x: (-x["avg_score"], x["total_tokens"]))

print(f"\n{'Strategy':<10} {'Size':<6} {'Overlap':<8} {'Chunks':<8} {'AvgLen':<8} {'Score':<8} {'Tokens':<8} {'Latency'}")
print("-" * 75)
for r in sorted_results:
    c = r["config"]
    print(f"{c['strategy']:<10} {c['chunk_size']:<6} {c['overlap']:<8} "
          f"{r['num_chunks']:<8} {r['avg_chunk_len']:<8} "
          f"{r['avg_score']:<8} {r['total_tokens']:<8} {r['avg_latency_ms']}ms")

best = sorted_results[0]
print(f"\n🏆 WINNER: strategy={best['config']['strategy']} | "
      f"chunk_size={best['config']['chunk_size']} | "
      f"score={best['avg_score']}")

# ── Key learnings ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY LEARNINGS FROM THIS EXPERIMENT")
print("=" * 70)
print("""
1. SMALL CHUNKS (200-500):
   + Best for simple fact retrieval (Q1 type)
   - Misses context for multi-sentence questions (Q2 type)
   - May split a complete thought across two chunks

2. LARGE CHUNKS (1500-2000):
   + Better for complex multi-part questions
   - More tokens per query = higher cost
   - Embedding quality drops (one vector for too many ideas)
   - May retrieve irrelevant sentences along with the relevant one

3. SENTENCE-AWARE vs FIXED:
   + Sentence-aware never cuts mid-thought
   + Better embedding quality (complete ideas)
   - Chunks vary in size (harder to predict token cost)

4. OVERLAP:
   + Prevents losing information at chunk boundaries
   + Critical for Q2-type questions that span sentences
   - Adds storage cost (duplicate content)

5. PRODUCTION RECOMMENDATION:
   - Start with 500-800 chars, sentence-aware, 10% overlap
   - Run this experiment on YOUR documents
   - The winner depends on your document structure, not a global rule
""")

# Save full results
with open("./outputs/week10/friday_chunk_experiment.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("Saved: outputs/week10/friday_chunk_experiment.json")
print("\n" + "=" * 70)
print("Friday Week 10 COMPLETE — Week 10 fully done!")
print("=" * 70)
