"""
Week 11 Wednesday — Groundedness Evaluation
Runs your RAG pipeline on 10 questions and measures
how grounded each answer is using LLM-as-judge.
"""
import os
from dotenv import load_dotenv

load_dotenv(), json, time, sys
sys.path.insert(0, "src/week11/wednesday_eval")

from groundedness_eval import evaluate_groundedness
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

os.makedirs("./outputs/week11", exist_ok=True)

AZURE_OAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OAI_KEY = os.getenv("AZURE_OPENAI_KEY")
SEARCH_ENDPOINT    = "https://fayz-search.search.windows.net"
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX       = "connectplus-rag"

oai_client = AzureOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version="2024-02-01"
)

# ── RAG pipeline ──────────────────────────────────────────────
def run_rag(question: str) -> tuple:
    """Returns (answer, context) for evaluation."""

    # Embed question
    vec = oai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    ).data[0].embedding

    # Search index
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_KEY)
    )
    results = search_client.search(
        search_text=None,
        vector_queries=[VectorizedQuery(
            vector=vec,
            k_nearest_neighbors=3,
            fields="embedding"
        )],
        select=["content"],
        top=3
    )
    chunks  = [r["content"] for r in results]
    context = "\n\n---\n\n".join(chunks)

    # Generate answer
    prompt = f"""You are Finn, a ConnectPlus UK customer service assistant.
Answer using ONLY information from the context below.
If the answer is not in the context say: I do not have that information.
Use British English. Maximum 3 sentences.

Context:
{context}

Question: {question}"""

    response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0
    )
    answer = response.choices[0].message.content
    return answer, context


# ── Load eval dataset ─────────────────────────────────────────
with open("src/week11/wednesday_eval/eval_dataset.json") as f:
    eval_dataset = json.load(f)

print("=" * 65)
print("WEEK 11 WEDNESDAY — Groundedness Evaluation")
print("=" * 65)
print(f"\nEvaluating {len(eval_dataset)} questions...")
print("Each question: RAG pipeline → LLM judge → groundedness score\n")

all_results = []
total_score = 0
category_scores = {}

for item in eval_dataset:
    qid      = item["id"]
    question = item["question"]
    category = item["category"]

    # Step 1 — Run RAG pipeline
    answer, context = run_rag(question)

    # Step 2 — Evaluate groundedness
    eval_result = evaluate_groundedness(question, context, answer)
    score       = eval_result.get("score", 0)
    total_score += score

    # Track by category
    if category not in category_scores:
        category_scores[category] = []
    category_scores[category].append(score)

    # Score emoji
    if score >= 5:
        emoji = "✅"
    elif score >= 4:
        emoji = "🟡"
    elif score >= 3:
        emoji = "🟠"
    else:
        emoji = "❌"

    print(f"{emoji} [{qid}] {category.upper()}")
    print(f"   Q: {question}")
    print(f"   A: {answer[:120]}")
    print(f"   Score: {score}/5 — {eval_result.get('reasoning', '')[:80]}")

    if eval_result.get("ungrounded_claims"):
        print(f"   ⚠️  Ungrounded: {eval_result['ungrounded_claims']}")
    print()

    all_results.append({
        "id":           qid,
        "category":     category,
        "question":     question,
        "answer":       answer,
        "context":      context[:300],
        "score":        score,
        "reasoning":    eval_result.get("reasoning", ""),
        "ungrounded":   eval_result.get("ungrounded_claims", []),
        "grounded":     eval_result.get("grounded_claims", []),
        "tokens":       eval_result.get("tokens_used", 0)
    })

    time.sleep(0.5)  # Avoid rate limiting

# ── Final report ──────────────────────────────────────────────
avg_score = total_score / len(eval_dataset)

print("=" * 65)
print("EVALUATION REPORT")
print("=" * 65)
print(f"\nOverall Groundedness Score: {avg_score:.2f} / 5.00")
print(f"Questions evaluated: {len(eval_dataset)}")

# Score distribution
score_counts = {1:0, 2:0, 3:0, 4:0, 5:0}
for r in all_results:
    score_counts[r["score"]] = score_counts.get(r["score"], 0) + 1

print(f"\nScore Distribution:")
for score in [5, 4, 3, 2, 1]:
    count = score_counts.get(score, 0)
    bar   = "█" * count
    label = {5:"Fully grounded",4:"Mostly grounded",
             3:"Partial",2:"Mostly ungrounded",1:"Not grounded"}[score]
    print(f"  {score} — {label:<20} {bar} ({count})")

print(f"\nScores by Category:")
for cat, scores in category_scores.items():
    avg = sum(scores) / len(scores)
    bar = "█" * int(avg)
    print(f"  {cat:<15} avg={avg:.1f} {bar}")

# Production threshold check
print(f"\nProduction Threshold Check:")
print(f"  Minimum acceptable score: 4.0/5.0")
if avg_score >= 4.0:
    print(f"  ✅ PASS — system is production ready ({avg_score:.2f})")
elif avg_score >= 3.0:
    print(f"  🟡 WARNING — needs improvement ({avg_score:.2f})")
else:
    print(f"  ❌ FAIL — do not deploy ({avg_score:.2f})")

print(f"""
What this score means in production:
  5.0 = Ship it. Every answer is grounded.
  4.0 = Acceptable. Minor phrasing issues only.
  3.0 = Needs work. Some hallucination detected.
  2.0 = Dangerous. Significant hallucination.
  1.0 = Do not deploy. System is making things up.

What to do if score is low:
  - Increase top_k (retrieve more chunks)
  - Reduce chunk size (more precise retrieval)
  - Strengthen system prompt grounding instructions
  - Add a validation layer (second GPT-4 call to verify)
""")

# Save full report
with open("./outputs/week11/wednesday_eval_results.json", "w") as f:
    json.dump({
        "overall_score":    round(avg_score, 3),
        "total_questions":  len(eval_dataset),
        "category_scores":  {k: round(sum(v)/len(v), 2)
                             for k, v in category_scores.items()},
        "results":          all_results
    }, f, indent=2)

print("Saved: outputs/week11/wednesday_eval_results.json")
print("\n" + "=" * 65)
print("Wednesday Week 11 COMPLETE")
print("=" * 65)
