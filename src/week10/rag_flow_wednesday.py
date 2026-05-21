import os, json, math, hashlib, time
from openai import AzureOpenAI

AZURE_OAI_ENDPOINT = "https://fayz-openai.openai.azure.com/"
AZURE_OAI_KEY      = "F2FBAVkbe8isc2gqXnSO7HYr4Gh03L8Y5FegiE4DM4yZi9NRfS03JQQJ99CEACYeBjFXJ3w3AAABACOGjfTg"
DEPLOYMENT_GPT     = "gpt-4o"
DEPLOYMENT_EMBED   = "text-embedding-3-small"
REFUSE_THRESHOLD   = 0.30

client = AzureOpenAI(
    azure_endpoint = AZURE_OAI_ENDPOINT,
    api_key        = AZURE_OAI_KEY,
    api_version    = "2024-02-01"
)
os.makedirs("./outputs/week10", exist_ok=True)

DOCUMENTS = [
    {"id":"cancel_p1", "source":"cancellation_policy.pdf", "page":1, "section":"Cancellation Terms",
     "content":"ConnectPlus cancellation requires 30 days written notice. No fee within first 14 days cooling off period. After 14 days 25 pound early termination fee for contracts under 12 months. After 12 months no termination fee. Equipment must be returned within 14 days or 75 pound non-return fee."},
    {"id":"cancel_p2", "source":"cancellation_policy.pdf", "page":2, "section":"Refund Policy",
     "content":"Refunds for overpayments and double billing charged twice processed within 5 to 7 working days to original payment method. Direct debit refunds take up to 10 working days. Disputed charges must be raised within 60 days."},
    {"id":"retention_p3", "source":"retention_playbook.pdf", "page":3, "section":"Risk Tiers",
     "content":"HIGH risk customers 3 or more support calls or explicit cancel intent: offer 3 month loyalty discount maximum 20 percent off monthly bill and assign dedicated Tier 2 agent priority engineer visit 48 hours. MEDIUM risk billing dispute: one month bill credit and free router upgrade. LOW risk: satisfaction survey and 20 pound referral bonus."},
    {"id":"products_p7", "source":"product_catalogue.pdf", "page":7, "section":"Broadband Plans",
     "content":"ConnectPlus broadband plans. Essential 35Mbps 25 pounds per month no minimum term. Standard 67Mbps 35 pounds per month 12 month contract. Premium 150Mbps 50 pounds per month 12 month contract. Ultrafast 500Mbps 70 pounds per month 24 month contract. Speed Promise exit without penalty if below 50 percent advertised speed for 3 consecutive days."},
    {"id":"billing_p2", "source":"billing_policy.pdf", "page":2, "section":"Billing Schedule",
     "content":"Bills generated 1st of each month. Payment due within 14 days. Late payment fee 12 pounds after 14 day grace period. Double billing charged twice in error refund within 3 to 5 working days. Direct debit failure 5 pound admin fee three consecutive failures account suspension."},
    {"id":"technical_p5", "source":"technical_support.pdf", "page":5, "section":"Common Issues",
     "content":"Internet dropping at specific times such as 9pm usually peak time congestion. Engineer visit to check line quality. Router restart hold reset button 10 seconds wait 2 minutes. Speed below 50 percent of plan log 3 consecutive days then call 0800 123 456 to invoke Speed Promise."},
    {"id":"billing_p3", "source":"billing_policy.pdf", "page":3, "section":"Price Increases",
     "content":"ConnectPlus must provide minimum 30 days written notice before any price increase. Customers may exit contract without penalty if they do not accept a price increase and notify ConnectPlus within 30 days of the notice."},
]

embedding_cache = {}

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    results, to_fetch, indices = [], [], []
    for i, t in enumerate(texts):
        key = hashlib.md5(t.lower().strip().encode()).hexdigest()
        if key in embedding_cache:
            results.append((i, embedding_cache[key]))
        else:
            to_fetch.append(t)
            indices.append(i)
    if to_fetch:
        r = client.embeddings.create(model=DEPLOYMENT_EMBED, input=to_fetch)
        for j, emb in enumerate(r.data):
            key = hashlib.md5(to_fetch[j].lower().strip().encode()).hexdigest()
            embedding_cache[key] = emb.embedding
            results.append((indices[j], emb.embedding))
    results.sort(key=lambda x: x[0])
    return [emb for _, emb in results]

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b)))

def bm25_score(query_terms, doc_text, k1=1.5, b=0.75, avg_len=60):
    doc_terms = doc_text.lower().split()
    score = 0.0
    for term in query_terms:
        tf = doc_terms.count(term.lower())
        if tf == 0:
            continue
        idf = math.log((len(doc_terms)+1)/(tf+0.5))
        tf_norm = (tf*(k1+1))/(tf+k1*(1-b+b*len(doc_terms)/avg_len))
        score += idf * tf_norm
    return score

def hybrid_search(query, docs, top_k=3):
    q_emb = embed(query)[0]
    vec_scores = {d["id"]: cosine_sim(q_emb, d["embedding"]) for d in docs}
    vector_ranked = sorted(vec_scores, key=vec_scores.get, reverse=True)
    terms = query.lower().split()
    kw_scored = sorted([(bm25_score(terms, d["content"]), i, d["id"]) for i,d in enumerate(docs)], reverse=True)
    keyword_ranked = [x[2] for x in kw_scored]
    rrf = {}
    for rank, doc_id in enumerate(vector_ranked):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1/(rank+60)
    for rank, doc_id in enumerate(keyword_ranked):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1/(rank+60)
    fused = sorted(rrf, key=rrf.get, reverse=True)
    id_map = {d["id"]: d for d in docs}
    return [(id_map[doc_id], vec_scores[doc_id]) for doc_id in fused[:top_k]]

RAG_SYSTEM = """You are a ConnectPlus customer service assistant.
Answer using ONLY information from the provided source documents.
If the answer is not in the documents say: I do not have that information. Please contact support@connectplus.co.uk
After every factual claim write the source in brackets like: [cancellation_policy.pdf, p.1]
Never invent figures, dates, or policies not present in the documents."""

def build_context(retrieved):
    context = ""
    for i, (doc, score) in enumerate(retrieved, 1):
        context += f"\n--- Document {i} [score: {score:.3f}] ---\n"
        context += f"Source: {doc['source']}, Page {doc['page']}\n"
        context += f"Content: {doc['content']}\n"
    return context

def rag_query(question, docs, conversation_history=None, validate=False):
    t0 = time.time()
    search_query = question
    if conversation_history:
        r = client.chat.completions.create(
            model=DEPLOYMENT_GPT,
            messages=[
                {"role":"system","content":"Rewrite the user question as a complete standalone search query. Return ONLY the rewritten query."},
                {"role":"user","content":f"Conversation:\n{conversation_history}\n\nQuestion: {question}\n\nStandalone query:"}
            ],
            max_tokens=60, temperature=0
        )
        search_query = r.choices[0].message.content.strip()
    retrieved = hybrid_search(search_query, docs, top_k=3)
    top_score = retrieved[0][1] if retrieved else 0
    if top_score < REFUSE_THRESHOLD:
        return {"question":question,"search_query":search_query,"answer":"I do not have that information. Please contact support@connectplus.co.uk","sources":[],"top_score":top_score,"refused":True,"latency_ms":round((time.time()-t0)*1000)}
    context = build_context(retrieved)
    t2 = time.time()
    messages = [{"role":"system","content":RAG_SYSTEM},
                {"role":"user","content":f"Customer question: {question}\n\n{context}"}]
    r = client.chat.completions.create(model=DEPLOYMENT_GPT, messages=messages, max_tokens=300, temperature=0)
    answer = r.choices[0].message.content
    gen_ms = round((time.time()-t2)*1000)
    validation = None
    if validate:
        has_citation = any(doc["source"].replace(".pdf","") in answer for doc,_ in retrieved)
        ctx_text = " ".join([d["content"] for d,_ in retrieved])
        vr = client.chat.completions.create(
            model=DEPLOYMENT_GPT,
            messages=[{"role":"user","content":f"Documents:\n{ctx_text}\n\nIs this answer supported by the documents? YES or NO only.\nAnswer: {answer}"}],
            max_tokens=5, temperature=0
        )
        validation = {"has_citation":has_citation,"is_grounded":"YES" in vr.choices[0].message.content.upper()}
    return {"question":question,"search_query":search_query,"answer":answer,"sources":[d["id"] for d,_ in retrieved],"top_score":round(top_score,4),"refused":False,"tokens":r.usage.total_tokens,"latency_ms":round((time.time()-t0)*1000),"gen_ms":gen_ms,"validation":validation}

print("="*60)
print("THE COMPLETE RAG FLOW — Wednesday Week 10")
print("="*60)

print("\n── Indexing documents ──")
embeddings = embed([d["content"] for d in DOCUMENTS])
for doc, emb in zip(DOCUMENTS, embeddings):
    doc["embedding"] = emb
print(f"  {len(DOCUMENTS)} documents indexed")

print("\n── TEST 1: Standard RAG queries ──\n")
standard_queries = [
    "What is the cancellation fee after 6 months?",
    "I was charged twice. When do I get my refund?",
    "My internet drops every night at 9pm.",
    "What is the best plan if I want 150Mbps?",
    "Can ConnectPlus raise my bill without warning?",
]
results = []
for q in standard_queries:
    r = rag_query(q, DOCUMENTS)
    print(f"Q: {q}")
    print(f"  Sources : {r['sources']}")
    print(f"  Score   : {r['top_score']} | Tokens: {r.get('tokens','N/A')} | Latency: {r['latency_ms']}ms")
    print(f"  Answer  : {r['answer'][:200]}\n")
    results.append(r)

print("── TEST 2: Out-of-domain refusal ──\n")
for q in ["What is the weather in London?", "Does ConnectPlus offer 5G mobile?", "How do I cook pasta?"]:
    r = rag_query(q, DOCUMENTS)
    status = "REFUSED" if r["refused"] else "ANSWERED"
    print(f"  [{status}] score={r['top_score']} — {q}")

print("\n── TEST 3: Query rewriting ──\n")
conversation = "Customer asked about cancellation. Agent explained 25 pound fee applies after 14 days."
followup = "What if I have been a customer for 2 years?"
r = rag_query(followup, DOCUMENTS, conversation_history=conversation)
print(f"  Original : {followup}")
print(f"  Rewritten: {r['search_query']}")
print(f"  Answer   : {r['answer'][:250]}")

print("\n── TEST 4: Answer validation ──\n")
for q in ["What is the cancellation fee?", "Does ConnectPlus offer satellite internet?"]:
    r = rag_query(q, DOCUMENTS, validate=True)
    v = r.get("validation") or {}
    print(f"  Q: {q}")
    print(f"  Refused: {r['refused']} | Citation: {v.get('has_citation')} | Grounded: {v.get('is_grounded')}")
    print(f"  Answer: {r['answer'][:150]}\n")

print("── TEST 5: Latency breakdown ──\n")
r = results[0]
print(f"  GPT generation : {r.get('gen_ms','N/A')}ms")
print(f"  Total          : {r['latency_ms']}ms")
print(f"  Cache entries  : {len(embedding_cache)}")

print("\n── TEST 6: Cost analysis ──\n")
answered = [r for r in results if not r.get("refused") and r.get("tokens")]
if answered:
    avg_tok = sum(r["tokens"] for r in answered) / len(answered)
    cost = avg_tok * 0.005 / 1000
    print(f"  Avg tokens/query : {avg_tok:.0f}")
    print(f"  Cost per query   : ${cost:.4f}")
    print(f"  1000 queries/day : ${cost*1000:.2f}/day")
    print(f"  Tip: gpt-4o-mini is 10x cheaper for simple factual RAG")

with open("./outputs/week10/wednesday_rag_flow.json","w") as f:
    json.dump(results, f, indent=2)

print("\n"+"="*60)
print("Wednesday Week 10 COMPLETE")
print("="*60)