"""
Week 11 Tuesday — Custom Python Tool Nodes
Runs all tools and shows how they connect to the RAG pipeline.
"""
import os
from dotenv import load_dotenv

load_dotenv(), json, sys, time
sys.path.insert(0, "src/week11/tuesday_tools")

from tool_time    import get_current_time
from tool_weather import get_weather
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

os.makedirs("./outputs/week11", exist_ok=True)

AZURE_OAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OAI_KEY = os.getenv("AZURE_OPENAI_KEY")
SEARCH_ENDPOINT    = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX       = "connectplus-rag"

oai_client = AzureOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version="2024-02-01"
)

print("=" * 60)
print("WEEK 11 TUESDAY — Custom Python Tool Nodes")
print("=" * 60)

# ── TEST 1: Time tool ─────────────────────────────────────────
print("\n── TOOL 1: get_current_time ──\n")
time_data = get_current_time()
for key, value in time_data.items():
    print(f"  {key}: {value}")

# ── TEST 2: Weather tool ──────────────────────────────────────
print("\n── TOOL 2: get_weather ──\n")
for city in ["london", "manchester", "addis ababa"]:
    result = get_weather(city)
    print(f"  {city.title()}: {result['summary']}")
    print(f"    Engineer visit: {result.get('engineer_visit', 'N/A')}\n")

# ── TEST 3: Full pipeline with tools ─────────────────────────
print("=" * 60)
print("FULL PIPELINE — Tools + RAG + GPT-4o")
print("=" * 60)

def search_knowledge_base(question: str) -> str:
    vec = oai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    ).data[0].embedding

    client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_KEY)
    )
    results = client.search(
        search_text=None,
        vector_queries=[VectorizedQuery(
            vector=vec,
            k_nearest_neighbors=3,
            fields="embedding"
        )],
        select=["content"],
        top=3
    )
    return "\n\n---\n\n".join([r["content"] for r in results])


def run_pipeline(question: str, city: str = "london") -> dict:
    """
    Full pipeline:
    Step 1 → get_current_time  (tool node)
    Step 2 → get_weather       (tool node)
    Step 3 → search_knowledge_base (RAG node)
    Step 4 → generate_answer   (LLM node)

    This is exactly what a Prompt Flow DAG executes.
    Each step feeds its output into the next step.
    """
    print(f"\nQ: {question}")

    # Tool nodes run first — they gather context
    t0         = time.time()
    time_data  = get_current_time()
    weather    = get_weather(city)
    context    = search_knowledge_base(question)
    tool_ms    = round((time.time() - t0) * 1000)

    # LLM node runs last — it uses all tool outputs
    prompt = f"""You are Finn, a ConnectPlus UK customer service assistant.

LIVE CONTEXT FROM TOOL NODES:
- Current UK time: {time_data['uk_time']}
- Today: {time_data['uk_date']}
- Support line: {time_data['support_status']}
- Weather in {city.title()}: {weather['summary']}
- Engineer visit today: {weather.get('engineer_visit', 'unknown')}

KNOWLEDGE BASE CONTEXT:
{context}

Answer the customer question using the above information.
Be specific — use the live time and weather data when relevant.
Use British English. Maximum 4 sentences."""

    t1      = time.time()
    response = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0
    )
    llm_ms = round((time.time() - t1) * 1000)
    answer = response.choices[0].message.content

    print(f"  A: {answer}")
    print(f"  Tools: {tool_ms}ms | LLM: {llm_ms}ms")

    return {
        "question":    question,
        "answer":      answer,
        "time_data":   time_data,
        "weather":     weather,
        "tool_ms":     tool_ms,
        "llm_ms":      llm_ms
    }

# Run 4 pipeline tests
tests = [
    {"question": "Can someone visit me today to fix my internet?",
     "city": "london"},
    {"question": "I want to cancel my contract. Is support open now?",
     "city": "manchester"},
    {"question": "What happens if my direct debit fails?",
     "city": "london"},
    {"question": "What broadband speeds do you offer?",
     "city": "london"},
]

results = []
for test in tests:
    result = run_pipeline(**test)
    results.append(result)

# Save results
with open("./outputs/week11/tuesday_tools_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 60)
print("WHAT YOU JUST BUILT — Senior Engineer Summary")
print("=" * 60)
print("""
Every AI assistant in production uses this exact pattern:

  Tool nodes run BEFORE the LLM:
    get_current_time → live date/time awareness
    get_weather      → live external API data
    search_kb        → live document retrieval

  LLM node runs LAST:
    It receives ALL tool outputs as context
    Its only job is language — tools do the data work

  Why this matters:
    GPT-4o alone cannot tell you the time
    GPT-4o alone cannot check the weather
    GPT-4o alone cannot search your documents
    Tools give it all three superpowers

  This is the foundation of Agents (Friday's lesson)
  An Agent is just this pattern made autonomous —
  the LLM decides WHICH tools to call and WHEN.
""")

print("Saved: outputs/week11/tuesday_tools_results.json")
print("Tuesday Week 11 COMPLETE")
print("=" * 60)
