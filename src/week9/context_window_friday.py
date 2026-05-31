import os
from dotenv import load_dotenv

load_dotenv()
import json
import tiktoken
from openai import AzureOpenAI
from datetime import datetime

AZURE_OAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OAI_KEY = os.getenv("AZURE_OPENAI_KEY")
DEPLOYMENT         = "gpt-4o"
TOKEN_LIMIT        = 3000   # artificially low to trigger trimming in demo
                             # in production use 100000

client = AzureOpenAI(
    azure_endpoint =  os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key        = "os.getenv("AZURE_OPENAI_KEY")",
    api_version    = "2024-02-01"
)

os.makedirs("./outputs/week9", exist_ok=True)
enc = tiktoken.encoding_for_model("gpt-4o")

SYSTEM = {
    "role": "system",
    "content": """You are Finn, ConnectPlus retention specialist.
Be concise — max 2 sentences. Remember everything the customer tells you.
If you recall information from a summary, use it naturally."""
}

# ── Core token counting ───────────────────────────────────────
def count_tokens(messages):
    total = 3
    for m in messages:
        total += 4
        total += len(enc.encode(m.get("content", "")))
        total += len(enc.encode(m.get("role", "")))
    return total

# ── Strategy 1: Sliding window ────────────────────────────────
def sliding_window(messages, max_turns=4):
    system  = [messages[0]]
    history = messages[1:]
    if len(history) > max_turns * 2:
        dropped = len(history) - max_turns * 2
        history = history[dropped:]
        print(f"  [Sliding window: dropped {dropped} oldest messages]")
    return system + history

# ── Strategy 2: Token-aware trimming ─────────────────────────
def token_aware_trim(messages, limit=TOKEN_LIMIT):
    result = list(messages)
    initial = count_tokens(result)
    drops   = 0
    while count_tokens(result) > limit and len(result) > 2:
        result.pop(1)
        drops += 1
    if drops:
        print(f"  [Token trim: dropped {drops} messages. {initial}→{count_tokens(result)} tokens]")
    return result

# ── Strategy 3: Summarisation ─────────────────────────────────
def summarise_and_compress(messages):
    if len(messages) <= 3:
        return messages
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages[1:]
    )
    print("  [Summarising conversation history...]")
    r = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content":
            f"Summarise this customer service conversation in 3 bullet points. "
            f"Include: main issue, offers made, customer sentiment.\n\n{history_text}"
        }],
        max_tokens=150, temperature=0
    )
    summary = r.choices[0].message.content
    compressed = [
        messages[0],
        {"role": "assistant",
         "content": f"[Conversation summary: {summary}]"}
    ]
    print(f"  [Compressed {len(messages)} messages → 2 messages with summary]")
    return compressed

def chat(messages, user_input):
    messages.append({"role": "user", "content": user_input})
    r = client.chat.completions.create(
        model=DEPLOYMENT, messages=messages,
        max_tokens=150, temperature=0.3
    )
    reply = r.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": reply})
    tokens = count_tokens(messages)
    return reply, tokens, messages

print("=" * 60)
print("CONTEXT WINDOW MANAGEMENT — Friday Week 9")
print("=" * 60)

results = {}

# ── DEMO 1: Sliding window ─────────────────────────────────────
print("\n── DEMO 1: Sliding window (max 4 turns) ──")
msgs1 = [SYSTEM.copy()]
exchanges = [
    "My name is Ahmed and my account is ACC-12345.",
    "I have had 6 support calls this month.",
    "My internet drops every night at 9pm.",
    "I work from home so this is affecting my job.",
    "What is my name?",   # ← tests if early info was dropped
]
for user_input in exchanges:
    reply, tokens, msgs1 = chat(msgs1, user_input)
    msgs1 = sliding_window(msgs1, max_turns=4)
    print(f"  User: {user_input}")
    print(f"  Finn: {reply}")
    print(f"  Context: {tokens} tokens, {len(msgs1)} messages\n")
results["sliding_window_final"] = reply

# ── DEMO 2: Token-aware trimming ──────────────────────────────
print("\n── DEMO 2: Token-aware trimming ──")
msgs2 = [SYSTEM.copy()]
for user_input in exchanges:
    reply, tokens, msgs2 = chat(msgs2, user_input)
    msgs2 = token_aware_trim(msgs2, limit=TOKEN_LIMIT)
    print(f"  User: {user_input}")
    print(f"  Finn: {reply}")
    print(f"  Context: {count_tokens(msgs2)} tokens\n")
results["token_trim_final"] = reply

# ── DEMO 3: Summarisation ─────────────────────────────────────
print("\n── DEMO 3: Summarisation strategy ──")
msgs3 = [SYSTEM.copy()]
for i, user_input in enumerate(exchanges):
    reply, tokens, msgs3 = chat(msgs3, user_input)
    print(f"  User: {user_input}")
    print(f"  Finn: {reply}")
    if count_tokens(msgs3) > TOKEN_LIMIT:
        msgs3 = summarise_and_compress(msgs3)
    print(f"  Context: {count_tokens(msgs3)} tokens, {len(msgs3)} messages\n")
results["summarisation_final"] = reply

# ── DEMO 4: Token counting deep dive ─────────────────────────
print("\n── DEMO 4: Token counting analysis ──")
test_messages = [
    SYSTEM,
    {"role": "user",      "content": "I want to cancel my subscription."},
    {"role": "assistant", "content": "I'm sorry to hear that. What's going on?"},
    {"role": "user",      "content": "I called support 6 times this month."},
]
total = count_tokens(test_messages)
print(f"  System message tokens:        {len(enc.encode(SYSTEM['content']))}")
print(f"  Full messages array tokens:   {total}")
print(f"  Per-message overhead (4×{len(test_messages)}): {4*len(test_messages)}")
print(f"  Remaining for response:       {128000 - total} tokens available")
print(f"  At 150 tokens/turn, turns before limit: {(128000 - total) // 300}")

# ── DEMO 5: Strategy comparison ───────────────────────────────
print("\n── DEMO 5: Strategy comparison ──")
print(f"  {'Strategy':<25} {'Messages':<12} {'Tokens':<10} {'Remembers name?'}")
print(f"  {'-'*60}")

name_q = "What is my name and account number?"

msgs_slide = sliding_window([SYSTEM.copy()] + [
    {"role": "user", "content": "My name is Ahmed, account ACC-12345."},
    {"role": "assistant", "content": "Got it Ahmed."},
    {"role": "user", "content": "Internet drops nightly."},
    {"role": "assistant", "content": "I see."},
    {"role": "user", "content": "Affects my work."},
    {"role": "assistant", "content": "Understood."},
], max_turns=1)

for strategy, msgs in [
    ("Sliding (max 1 turn)", msgs_slide),
    ("Full history",         [SYSTEM, 
        {"role": "user", "content": "My name is Ahmed, account ACC-12345."},
        {"role": "assistant", "content": "Got it Ahmed."},
        {"role": "user", "content": name_q}])
]:
    r = client.chat.completions.create(
        model=DEPLOYMENT, messages=msgs + [{"role":"user","content":name_q}],
        max_tokens=50, temperature=0
    )
    answer = r.choices[0].message.content.strip()
    tok = count_tokens(msgs)
    remembers = "YES" if "ahmed" in answer.lower() or "acc" in answer.lower() else "NO ← context lost"
    print(f"  {strategy:<25} {len(msgs):<12} {tok:<10} {remembers}")
    print(f"    Response: {answer[:80]}")

with open("./outputs/week9/friday_context_results.json","w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("Friday Week 9 COMPLETE — Week 9 done.")
