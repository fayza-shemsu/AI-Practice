import os
from dotenv import load_dotenv

load_dotenv()
import json
import time
import requests

SEARCH_ENDPOINT = "https://fayz-search.search.windows.net"
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME      = "connectplus-contracts"
API_VERSION     = "2024-07-01"

os.makedirs("./outputs/week8", exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# STEP 1 — Delete index if it exists (clean slate for learning)
# ─────────────────────────────────────────────────────────────────
def delete_index_if_exists():
    response = requests.delete(
        f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION}",
        headers={"api-key": SEARCH_KEY}
    )
    if response.status_code == 204:
        print(f"  Deleted existing index: {INDEX_NAME}")
    elif response.status_code == 404:
        print(f"  No existing index to delete")
    else:
        print(f"  Delete response: {response.status_code}")


# ─────────────────────────────────────────────────────────────────
# STEP 2 — Define and create the Index schema
# This is permanent — design it carefully in production
# ─────────────────────────────────────────────────────────────────
def create_index():
    print("\nCreating index schema...")

    schema = {
        "name": INDEX_NAME,
        "fields": [
            # Every index must have exactly one key field — string type
            {
                "name": "id",
                "type": "Edm.String",
                "key": True,
                "retrievable": True,
                "searchable": False,  # IDs are never searched by keyword
                "filterable": True,
                "sortable": False
            },
            # The main text content — full inverted index built on this
            {
                "name": "content",
                "type": "Edm.String",
                "key": False,
                "retrievable": True,
                "searchable": True,   # keyword search enabled
                "filterable": False,  # too long for filter
                "sortable": False,    # too long for sort
                "analyzer": "en.microsoft"  # language-aware tokenizer
            },
            # Document title — searchable and retrievable
            {
                "name": "title",
                "type": "Edm.String",
                "retrievable": True,
                "searchable": True,
                "filterable": True,
                "sortable": True,
                "analyzer": "en.microsoft"
            },
            # Structured fields — filterable but not full-text searchable
            {
                "name": "contract_number",
                "type": "Edm.String",
                "retrievable": True,
                "searchable": True,
                "filterable": True,
                "sortable": True
            },
            {
                "name": "customer_name",
                "type": "Edm.String",
                "retrievable": True,
                "searchable": True,
                "filterable": True,
                "sortable": True
            },
            {
                "name": "plan_name",
                "type": "Edm.String",
                "retrievable": True,
                "searchable": True,
                "filterable": True,
                "sortable": True
            },
            {
                "name": "monthly_fee",
                "type": "Edm.Double",
                "retrievable": True,
                "searchable": False,  # numbers are not keyword-searched
                "filterable": True,   # filter: monthly_fee gt 50
                "sortable": True      # sort: cheapest to most expensive
            },
            {
                "name": "contract_duration_months",
                "type": "Edm.Int32",
                "retrievable": True,
                "searchable": False,
                "filterable": True,
                "sortable": True
            },
            {
                "name": "start_date",
                "type": "Edm.DateTimeOffset",
                "retrievable": True,
                "searchable": False,
                "filterable": True,
                "sortable": True
            },
            {
                "name": "customer_country",
                "type": "Edm.String",
                "retrievable": True,
                "searchable": False,
                "filterable": True,
                "sortable": True
            },
        ]
    }

    response = requests.put(
        f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION}",
        headers={
            "api-key": SEARCH_KEY,
            "Content-Type": "application/json"
        },
        json=schema
    )

    if response.status_code in (200, 201):
        print(f"  Index created: {INDEX_NAME}")
        print(f"  Fields defined: {len(schema['fields'])}")
    else:
        raise Exception(f"Index creation failed: {response.status_code} {response.text}")

    return schema


# ─────────────────────────────────────────────────────────────────
# STEP 3 — Upload documents to the index
# In production this comes from your Document Intelligence pipeline
# ─────────────────────────────────────────────────────────────────
def upload_documents():
    print("\nUploading documents to index...")

    documents = [
        {
            "id": "CNT-2026-0847",
            "title": "Service Agreement — Ahmed Al-Rashidi",
            "content": "ConnectPlus Pro Plan monthly fee EUR 49.99 contract duration 24 months termination fee EUR 199.00 start date 2026-05-01 customer Ahmed Al-Rashidi address 456 Client Avenue Dubai UAE priority support add-on SMS notification pack",
            "contract_number": "CNT-2026-0847",
            "customer_name": "Ahmed Al-Rashidi",
            "plan_name": "ConnectPlus Pro",
            "monthly_fee": 49.99,
            "contract_duration_months": 24,
            "start_date": "2026-05-01T00:00:00Z",
            "customer_country": "UAE"
        },
        {
            "id": "CNT-2026-0901",
            "title": "Service Agreement — Sara Mensah",
            "content": "ConnectPlus Basic Plan monthly fee EUR 29.99 contract duration 12 months termination fee EUR 99.00 start date 2026-05-03 customer Sara Mensah address 12 Palm Road Accra Ghana short term contract basic tier",
            "contract_number": "CNT-2026-0901",
            "customer_name": "Sara Mensah",
            "plan_name": "ConnectPlus Basic",
            "monthly_fee": 29.99,
            "contract_duration_months": 12,
            "start_date": "2026-05-03T00:00:00Z",
            "customer_country": "Ghana"
        },
        {
            "id": "CNT-2026-0955",
            "title": "Service Agreement — Luca Bianchi",
            "content": "ConnectPlus Enterprise Plan monthly fee EUR 79.99 contract duration 36 months termination fee EUR 299.00 start date 2026-04-15 customer Luca Bianchi address Via Roma 88 Milan Italy enterprise tier long term commitment",
            "contract_number": "CNT-2026-0955",
            "customer_name": "Luca Bianchi",
            "plan_name": "ConnectPlus Enterprise",
            "monthly_fee": 79.99,
            "contract_duration_months": 36,
            "start_date": "2026-04-15T00:00:00Z",
            "customer_country": "Italy"
        },
        {
            "id": "CNT-2026-1002",
            "title": "Service Agreement — Yuki Tanaka",
            "content": "ConnectPlus Pro Plan monthly fee EUR 49.99 contract duration 24 months termination fee EUR 199.00 start date 2026-04-20 customer Yuki Tanaka address 3-5 Shibuya Tokyo Japan pro tier standard commitment",
            "contract_number": "CNT-2026-1002",
            "customer_name": "Yuki Tanaka",
            "plan_name": "ConnectPlus Pro",
            "monthly_fee": 49.99,
            "contract_duration_months": 24,
            "start_date": "2026-04-20T00:00:00Z",
            "customer_country": "Japan"
        },
        {
            "id": "CNT-2026-1078",
            "title": "Service Agreement — Fatima Al-Zahra",
            "content": "ConnectPlus Business Plan monthly fee EUR 59.99 contract duration 24 months termination fee EUR 199.00 start date 2026-05-01 customer Fatima Al-Zahra address King Fahd Road Riyadh Saudi Arabia business tier",
            "contract_number": "CNT-2026-1078",
            "customer_name": "Fatima Al-Zahra",
            "plan_name": "ConnectPlus Business",
            "monthly_fee": 59.99,
            "contract_duration_months": 24,
            "start_date": "2026-05-01T00:00:00Z",
            "customer_country": "Saudi Arabia"
        },
    ]

    batch = {"value": [{"@search.action": "upload", **doc} for doc in documents]}

    response = requests.post(
        f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/index?api-version={API_VERSION}",
        headers={
            "api-key": SEARCH_KEY,
            "Content-Type": "application/json"
        },
        json=batch
    )

    if response.status_code == 200:
        results = response.json().get("value", [])
        succeeded = sum(1 for r in results if r.get("status"))
        print(f"  Documents uploaded: {succeeded}/{len(documents)}")
        for r in results:
            status = "OK" if r.get("status") else "FAIL"
            print(f"  [{status}] {r.get('key')}")
    else:
        raise Exception(f"Upload failed: {response.status_code} {response.text}")


# ─────────────────────────────────────────────────────────────────
# STEP 4 — Run search queries demonstrating all capabilities
# ─────────────────────────────────────────────────────────────────
def run_searches():
    print("\n" + "=" * 65)
    print("SEARCH DEMONSTRATIONS")
    print("=" * 65)

    def search(label, params):
        print(f"\n--- {label} ---")
        response = requests.get(
            f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs",
            headers={"api-key": SEARCH_KEY},
            params={"api-version": API_VERSION, **params}
        )
        data = response.json()
        results = data.get("value", [])
        count = data.get("@odata.count")
        if count is not None:
            print(f"  Total matches: {count}")
        for r in results:
            score = r.get("@search.score", 0)
            print(f"  [{score:.3f}] {r.get('customer_name'):<20} {r.get('plan_name'):<25} EUR {r.get('monthly_fee')}")
        return results

    # 1. Full-text keyword search — BM25 relevance scoring
    search("1. Full-text: search for 'enterprise long term'",
        {"search": "enterprise long term", "select": "customer_name,plan_name,monthly_fee"})

    # 2. Filter — exact structured match, no keyword scoring
    search("2. Filter: monthly_fee less than 50",
        {"search": "*", "filter": "monthly_fee lt 50",
         "select": "customer_name,plan_name,monthly_fee", "$count": "true"})

    # 3. Combined search + filter — keyword in content AND price constraint
    search("3. Search + Filter: 'pro plan' AND monthly_fee lt 55",
        {"search": "pro plan", "filter": "monthly_fee lt 55",
         "select": "customer_name,plan_name,monthly_fee"})

    # 4. Facets — aggregate counts by field value (used for sidebar filters)
    print("\n--- 4. Facets: count by plan_name ---")
    response = requests.get(
        f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs",
        headers={"api-key": SEARCH_KEY},
        params={"api-version": API_VERSION, "search": "*",
                "facet": "plan_name", "facet": "customer_country"}
    )
    facets = response.json().get("@search.facets", {})
    for field, buckets in facets.items():
        print(f"  {field}:")
        for b in buckets:
            print(f"    {b['value']:<30} count={b['count']}")

    # 5. Sort by monthly_fee ascending
    search("5. Sort: cheapest contracts first",
        {"search": "*", "orderby": "monthly_fee asc",
         "select": "customer_name,plan_name,monthly_fee"})

    # 6. Top N with pagination
    search("6. Pagination: top 2 results, skip 1",
        {"search": "*", "top": "2", "skip": "1",
         "select": "customer_name,plan_name,monthly_fee"})

    # 7. Select specific fields only
    search("7. Projection: only name and country",
        {"search": "*", "select": "customer_name,customer_country"})


# ─────────────────────────────────────────────────────────────────
# STEP 5 — Save index stats
# ─────────────────────────────────────────────────────────────────
def save_index_stats():
    response = requests.get(
        f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/stats?api-version={API_VERSION}",
        headers={"api-key": SEARCH_KEY}
    )
    stats = response.json()
    doc_count  = stats.get("documentCount", 0)
    index_size = stats.get("storageSize", 0)
    print(f"\nIndex stats:")
    print(f"  Documents indexed: {doc_count}")
    print(f"  Index size:        {index_size:,} bytes")

    summary = {
        "index_name":   INDEX_NAME,
        "document_count": doc_count,
        "storage_bytes":  index_size,
        "fields": [
            {"name": "id",             "type": "Edm.String",        "key": True},
            {"name": "content",        "type": "Edm.String",        "searchable": True},
            {"name": "title",          "type": "Edm.String",        "searchable": True},
            {"name": "contract_number","type": "Edm.String",        "filterable": True},
            {"name": "customer_name",  "type": "Edm.String",        "filterable": True},
            {"name": "plan_name",      "type": "Edm.String",        "filterable": True, "facetable": True},
            {"name": "monthly_fee",    "type": "Edm.Double",        "sortable": True, "filterable": True},
            {"name": "start_date",     "type": "Edm.DateTimeOffset","sortable": True},
            {"name": "customer_country","type": "Edm.String",       "filterable": True, "facetable": True},
        ]
    }
    path = "./outputs/week8/search_index_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {path}")


# ─────────────────────────────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("AZURE AI SEARCH SETUP — Week 8 Wednesday")
print("=" * 65)

delete_index_if_exists()
schema = create_index()
upload_documents()
time.sleep(2)  # let indexing complete
run_searches()
save_index_stats()

print()
print("=" * 65)
print("Wednesday Week 8 COMPLETE")
print("  Index created with 9 fields")
print("  5 contracts indexed")
print("  7 search patterns demonstrated")
print("=" * 65)
