import os
import json
import time
import requests

SEARCH_ENDPOINT  = "https://fayz-search.search.windows.net"
SEARCH_KEY       = "P8vmYuqS7rOctpch0i8SMVFjOBokUtCpufq9B1s9cmAzSeCFyHJC"
STORAGE_CONN     = "DefaultEndpointsProtocol=https;AccountName=aiinterns;AccountKey=Fj6QB7KPLoNaJVqLwUqcXx2QIqTslcUG8ygVMBAzidaBHm4HrBmKE73wfbjpWRhPz8r6fJ3Bx1W1+ASt9X/49w==;EndpointSuffix=core.windows.net"
INDEX_NAME       = "fifty-contracts-index"
DATASOURCE_NAME  = "fifty-contracts-datasource"
SKILLSET_NAME    = "fifty-contracts-skillset"
INDEXER_NAME     = "fifty-contracts-indexer"
CONTAINER_NAME   = "fifty-contracts"
API_VERSION      = "2024-07-01"

os.makedirs("./outputs/week8", exist_ok=True)
HEADERS = {"api-key": SEARCH_KEY, "Content-Type": "application/json"}

def delete_if_exists(rtype, name):
    r = requests.delete(
        f"{SEARCH_ENDPOINT}/{rtype}/{name}?api-version={API_VERSION}",
        headers=HEADERS)
    print(f"  Deleted {name}" if r.status_code == 204 else f"  Not found: {name}")

def create_index():
    print("\nCreating index...")
    schema = {
        "name": INDEX_NAME,
        "fields": [
            {"name": "id",                 "type": "Edm.String",           "key": True,  "retrievable": True,  "searchable": False, "filterable": True},
            {"name": "content",            "type": "Edm.String",           "key": False, "retrievable": True,  "searchable": True,  "filterable": False, "analyzer": "en.microsoft"},
            {"name": "metadata_storage_name","type": "Edm.String",         "retrievable": True,  "searchable": False, "filterable": True,  "sortable": True},
            {"name": "metadata_storage_last_modified","type": "Edm.DateTimeOffset","retrievable": True,"searchable": False,"filterable": True,"sortable": True},
            {"name": "keyphrases",         "type": "Collection(Edm.String)","retrievable": True, "searchable": True,  "filterable": True},
            {"name": "persons",            "type": "Collection(Edm.String)","retrievable": True, "searchable": True,  "filterable": True},
            {"name": "locations",          "type": "Collection(Edm.String)","retrievable": True, "searchable": True,  "filterable": True},
            {"name": "language",           "type": "Edm.String",           "retrievable": True,  "searchable": False, "filterable": True},
        ]
    }
    r = requests.put(
        f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION}",
        headers=HEADERS, json=schema)
    print(f"  Index: {r.status_code}")

def create_datasource():
    print("\nCreating datasource...")
    ds = {
        "name": DATASOURCE_NAME,
        "type": "azureblob",
        "credentials": {"connectionString": STORAGE_CONN},
        "container": {"name": CONTAINER_NAME, "query": ""},
        "dataChangeDetectionPolicy": {
            "@odata.type": "#Microsoft.Azure.Search.HighWaterMarkChangeDetectionPolicy",
            "highWaterMarkColumnName": "metadata_storage_last_modified"
        }
    }
    r = requests.put(
        f"{SEARCH_ENDPOINT}/datasources/{DATASOURCE_NAME}?api-version={API_VERSION}",
        headers=HEADERS, json=ds)
    print(f"  Datasource: {r.status_code}")

def create_skillset():
    print("\nCreating skillset...")
    ss = {
        "name": SKILLSET_NAME,
        "description": "Enrich 50 contracts",
        "skills": [
            {
                "@odata.type": "#Microsoft.Skills.Text.LanguageDetectionSkill",
                "context": "/document",
                "inputs":  [{"name": "text",         "source": "/document/content"}],
                "outputs": [{"name": "languageCode",  "targetName": "language"}]
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.KeyPhraseExtractionSkill",
                "context": "/document",
                "maxKeyPhraseCount": 15,
                "inputs":  [{"name": "text", "source": "/document/content"},
                            {"name": "languageCode", "source": "/document/language"}],
                "outputs": [{"name": "keyPhrases", "targetName": "keyphrases"}]
            },
            {
                "@odata.type": "#Microsoft.Skills.Text.V3.EntityRecognitionSkill",
                "context": "/document",
                "categories": ["Person", "Location"],
                "inputs":  [{"name": "text", "source": "/document/content"},
                            {"name": "languageCode", "source": "/document/language"}],
                "outputs": [{"name": "persons",   "targetName": "persons"},
                            {"name": "locations", "targetName": "locations"}]
            }
        ]
    }
    r = requests.put(
        f"{SEARCH_ENDPOINT}/skillsets/{SKILLSET_NAME}?api-version={API_VERSION}",
        headers=HEADERS, json=ss)
    print(f"  Skillset: {r.status_code}")

def create_and_run_indexer():
    print("\nCreating and running indexer...")
    idxr = {
        "name":            INDEXER_NAME,
        "dataSourceName":  DATASOURCE_NAME,
        "skillsetName":    SKILLSET_NAME,
        "targetIndexName": INDEX_NAME,
        "parameters": {
            "batchSize": 10,
            "maxFailedItems": 10,
            "configuration": {
                "dataToExtract": "contentAndMetadata",
                "parsingMode":   "default"
            }
        },
        "fieldMappings": [
            {"sourceFieldName": "metadata_storage_path",
             "targetFieldName": "id",
             "mappingFunction": {"name": "base64Encode"}},
            {"sourceFieldName": "metadata_storage_name",
             "targetFieldName": "metadata_storage_name"},
            {"sourceFieldName": "metadata_storage_last_modified",
             "targetFieldName": "metadata_storage_last_modified"},
        ],
        "outputFieldMappings": [
            {"sourceFieldName": "/document/language",   "targetFieldName": "language"},
            {"sourceFieldName": "/document/keyphrases", "targetFieldName": "keyphrases"},
            {"sourceFieldName": "/document/persons",    "targetFieldName": "persons"},
            {"sourceFieldName": "/document/locations",  "targetFieldName": "locations"},
        ]
    }
    r = requests.put(
        f"{SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}?api-version={API_VERSION}",
        headers=HEADERS, json=idxr)
    print(f"  Indexer created: {r.status_code}")

    requests.post(
        f"{SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}/run?api-version={API_VERSION}",
        headers=HEADERS)
    print("  Indexer running...")

    for i in range(30):
        time.sleep(5)
        status = requests.get(
            f"{SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}/status?api-version={API_VERSION}",
            headers=HEADERS).json()
        last = status.get("lastResult", {})
        s    = last.get("status", "unknown")
        proc = last.get("itemsProcessed", 0)
        fail = last.get("itemsFailed", 0)
        print(f"  [{i+1:02d}] {s}  processed={proc}  failed={fail}")
        if s == "success":
            print(f"  Indexer complete — {proc} documents indexed")
            return proc
        elif s in ("error", "failed"):
            print(f"  Errors: {last.get('errors', [])}")
            return proc
    return 0


# ─────────────────────────────────────────────────────────────────
# THE MAIN DELIVERABLE — 7 search patterns on 50 real PDFs
# ─────────────────────────────────────────────────────────────────
def run_all_searches():
    print("\n" + "=" * 65)
    print("SEARCHING 50 CONTRACTS — All Query Patterns")
    print("=" * 65)
    results_log = []

    def search(label, params, extra_params=None):
        print(f"\n{'─'*65}")
        print(f"QUERY: {label}")
        print(f"{'─'*65}")
        p = {"api-version": API_VERSION, **params}
        if extra_params:
            p.update(extra_params)
        r = requests.get(
            f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs",
            headers={"api-key": SEARCH_KEY},
            params=p)
        data = r.json()
        hits = data.get("value", [])
        count = data.get("@odata.count")
        if count:
            print(f"Total matches: {count}")
        print(f"Results returned: {len(hits)}")
        for doc in hits:
            score = doc.get("@search.score", 0)
            name  = doc.get("metadata_storage_name", "?")
            kp    = doc.get("keyphrases", [])[:4]
            p_    = doc.get("persons",   [])
            loc   = doc.get("locations", [])
            lang  = doc.get("language",  "")
            print(f"  [{score:.3f}] {name:<30} lang={lang}")
            if kp:  print(f"    keyphrases: {kp}")
            if p_:  print(f"    persons:    {p_}")
            if loc: print(f"    locations:  {loc}")
        results_log.append({"query": label, "count": len(hits),
                            "top_result": hits[0].get("metadata_storage_name") if hits else None})
        return hits

    # ── 1. Simple keyword search ─────────────────────────────────
    search("Keyword: termination fee",
        {"search": "termination fee", "top": "5",
         "select": "metadata_storage_name,keyphrases,persons,language"})

    # ── 2. Fuzzy search — 1 edit distance ────────────────────────
    # ~1 means match words within 1 character change
    # "terminaton" → finds "termination"
    # "montly"     → finds "monthly"
    # "enterprize" → finds "enterprise"
    search("Fuzzy~1: terminaton fee (typo in termination)",
        {"search": "terminaton~1 fee", "top": "5",
         "select": "metadata_storage_name,keyphrases,persons,language"})

    search("Fuzzy~1: montly enterprize (two typos)",
        {"search": "montly~1 enterprize~1", "top": "5",
         "select": "metadata_storage_name,keyphrases,persons,language"})

    search("Fuzzy~2: cancllation (2 edits from cancellation)",
        {"search": "cancllation~2", "top": "5",
         "select": "metadata_storage_name,keyphrases,persons,language"})

    # ── 3. Phrase search — exact word order ──────────────────────
    # "early termination" as a phrase, not individual words
    search('Phrase: "early termination" (exact phrase)',
        {"search": '"early termination"', "top": "5",
         "select": "metadata_storage_name,keyphrases,persons,language"})

    # ── 4. Boosting — weight one term more than another ──────────
    # enterprise^3 means triple the score for documents with "enterprise"
    search("Boosted: enterprise^3 OR basic^1 (enterprise weighted 3x)",
        {"search": "enterprise^3 OR basic^1", "top": "5",
         "select": "metadata_storage_name,keyphrases,persons,language"})

    # ── 5. Filter + keyword ──────────────────────────────────────
    search("Filter + Keyword: search=pro AND location contains Japan",
        {"search": "pro plan", "top": "5",
         "filter": "locations/any(l: l eq 'Japan')",
         "select": "metadata_storage_name,keyphrases,persons,locations,language"})

    # ── 6. Facets on 50 documents ────────────────────────────────
    print(f"\n{'─'*65}")
    print("FACETS: Distribution across 50 contracts")
    print(f"{'─'*65}")
    r = requests.get(
        f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs",
        headers={"api-key": SEARCH_KEY},
        params=[("api-version", API_VERSION), ("search", "*"),
                ("facet", "language"), ("facet", "locations"),
                ("top", "0")])
    facets = r.json().get("@search.facets", {})
    for field, buckets in facets.items():
        print(f"  {field}:")
        for b in (buckets or [])[:10]:
            bar = "█" * b["count"]
            print(f"    {b['value']:<25} {bar} ({b['count']})")

    # ── 7. Person search across all 50 docs ──────────────────────
    search("Person filter: contracts involving Japan or Italy",
        {"search": "*", "top": "10",
         "filter": "locations/any(l: l eq 'Japan') or locations/any(l: l eq 'Italy')",
         "select": "metadata_storage_name,persons,locations,language"})

    return results_log


# ─────────────────────────────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("FRIDAY WEEK 8 — Search 50 PDFs with Fuzzy + Advanced Queries")
print("=" * 65)

print("\nCleaning up...")
delete_if_exists("indexers",    INDEXER_NAME)
delete_if_exists("skillsets",   SKILLSET_NAME)
delete_if_exists("datasources", DATASOURCE_NAME)
delete_if_exists("indexes",     INDEX_NAME)

create_index()
create_datasource()
create_skillset()
docs_indexed = create_and_run_indexer()

time.sleep(5)
results_log = run_all_searches()

summary = {
    "total_pdfs_indexed": docs_indexed,
    "index_name":         INDEX_NAME,
    "query_patterns_tested": [r["query"] for r in results_log],
    "results_per_query":     {r["query"]: r["count"] for r in results_log},
}
with open("./outputs/week8/friday_search_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 65)
print("FRIDAY WEEK 8 COMPLETE")
print("=" * 65)
print(json.dumps(summary, indent=2))
