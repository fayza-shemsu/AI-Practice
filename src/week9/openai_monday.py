import os
from dotenv import load_dotenv

load_dotenv()

import json

from openai import AzureOpenAI

AZURE_OAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_OAI_KEY = os.getenv("AZURE_OPENAI_KEY")

DEPLOYMENT_35      = "gpt-4o"

DEPLOYMENT_4O      = "gpt-4o"

client = AzureOpenAI(

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),

    api_key        = "B2W56Y0rYjtG79LbyNLyIVNJNaeR7ESba195PkOnIhv2MvCy409WJQQJ99CEACYeBjFXJ3w3AAABACOGrPCG",

    api_version    = "2024-02-01"

)

os.makedirs("./outputs/week9", exist_ok=True)

results = {}

print("=" * 60)

print("AZURE OPENAI — Monday Week 9")

print("=" * 60)

# Test 1 — basic call to gpt-35-turbo

print("\nTest 1 — gpt-35-turbo basic call")

r1 = client.chat.completions.create(

    model       = DEPLOYMENT_35,

    messages    = [{"role": "user", "content": "What is customer churn in telecoms? Answer in 2 sentences."}],

    max_tokens  = 200,

    temperature = 0.7,

)

print(f"Response: {r1.choices[0].message.content}")

print(f"Tokens used: {r1.usage.total_tokens}")

print(f"Finish reason: {r1.choices[0].finish_reason}")

results["test1"] = r1.choices[0].message.content

# Test 2 — gpt-4o with your real RAI data

print("\nTest 2 — gpt-4o with your real project data")

r2 = client.chat.completions.create(

    model       = DEPLOYMENT_4O,

    messages    = [{

        "role": "user",

        "content": (

            "I built a churn model. SHAP analysis shows:\n"

            "1. Support Calls: 0.1532 importance (top driver)\n"

            "2. Tenure: 0.1133 (new customers churn more)\n"

            "3. Gender: 0.0905 (fairness concern — 6.6% accuracy gap)\n\n"

            "What are the 3 most important actions before deploying to production?"

        )

    }],

    max_tokens  = 400,

    temperature = 0.5,

)

print(f"Response:\n{r2.choices[0].message.content}")

print(f"Tokens used: {r2.usage.total_tokens}")

results["test2"] = r2.choices[0].message.content

# Test 3 — temperature comparison

print("\nTest 3 — temperature effect (0.0 vs 1.0)")

for temp in [0.0, 1.0]:

    r = client.chat.completions.create(

        model       = DEPLOYMENT_35,

        messages    = [{"role": "user", "content": "Give one creative retention offer for a churning customer. One sentence only."}],

        max_tokens  = 60,

        temperature = temp,

    )

    print(f"  temp={temp}: {r.choices[0].message.content.strip()}")

# Test 4 — finish_reason check

print("\nTest 4 — finish_reason (max_tokens too low)")

r4 = client.chat.completions.create(

    model      = DEPLOYMENT_35,

    messages   = [{"role": "user", "content": "Explain churn prediction in 5 sentences."}],

    max_tokens = 25,

    temperature= 0.5,

)

print(f"  finish_reason: {r4.choices[0].finish_reason}")

print(f"  response: {r4.choices[0].message.content}")

if r4.choices[0].finish_reason == "length":

    print("  WARNING: Response was cut off — increase max_tokens")

with open("./outputs/week9/monday_results.json", "w") as f:

    json.dump(results, f, indent=2)

print()

print("=" * 60)

print("Monday Week 9 COMPLETE")

