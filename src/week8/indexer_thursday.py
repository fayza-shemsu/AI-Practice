import os
from dotenv import load_dotenv

load_dotenv()
import json
import time
import requests

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
STORAGE_CONN    = "os.getenv("AZURE_STORAGE_CONNECTION_STRING")"
CONTAINER_NAME  = "training-forms"
INDEX_NAME      = "connectplus-blob-index"
DATASOURCE_NAME = "connectplus-blob-datasource"
SKILLSET_NAME   = "connectplus-skillset"
INDEXER_NAME    = "connectplus-indexer"
API_VERSION     = "2024-07-01"

os.makedirs("./outputs/week8", exist_ok=True)

HEADERS = {
    "api-key": SEARCH_KEY,
    "Content-Type": "application/json"
}

def delete_if_exists(resource_type, name):
    url = f"{SEARCH_ENDPOINT}/{resource_type}/{name}?api-version={API_VERSION}"
    r = requests.delete(url, headers=HEADERS)
    if r.status_code == 204:
        print(f"  Deleted: {name}")
    elif r.status_code == 404:
        print(f"  Not found (OK): {name}")


# ─────────────────────────────────────────────────────────────────
# STEP 1 — Create the Index
# This defines what fields the indexer will populate
# ─────────────────────────────────────────────────────────────────
def create_index():
    print("\nStep 1: Creating index...")
    schema = {
        "name": INDEX_NAME,
        "fields": [
            {"name": "id",               "type": "Edm.String",  "key": True,  "retrievable": True, "searchable": False, "filterable": True},
            {"name": "content",          "type": "Edm.String",  "key": False, "retrievable": True, "searchable": True,  "filterable": False, "analyzer": "en.microsoft"},
            {"name": "metadata_title",   "type": "Edm.String",  "retrievable": True, "searchable": True,  "filterable": True,  "analyzer": "en.microsoft"},
            {"name": "metadata_storage_name", "type": "Edm.String", "retrievable": True, "searchable": False, "filterable": True},
            {"name": "metadata_storage_path", "type": "Edm.String", "retrievable": True, "searchable": False, "filterable": False},
            {"name": "metadata_storage_size", "type": "Edm.Int64",  "retrievable": True, "searchable": False, "filterable": True,  "sortable": True},
            {"name": "metadata_storage_last_modified", "type": "Edm.DateTimeOffset", "retrievable": True, "searchable": False, "filterable": True, "sortable": True},
            {"name": "keyphrases",       "type": "Collection(Edm.String)", "retrievable": True, "searchable": True,  "filterable": True},
            {"name": "language",         "type": "Edm.String",  "retrievable": True, "searchable": False, "filterable": True},
            {"name": "organizations",    "type": "Collection(Edm.String)", "retrievable": True, "searchable": True,  "filterable": True},
            {"name": "persons",          "type": "Collection(Edm.String)", "retrievable": True, "searchable": True,  "filterable": True},
            {"name": "locations",        "type": "Collection(Edm.String)", "retrievable": True, "searchable": True,  "filterable": True},
        ]
    }
    r = requests.put(
        f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version={API_VERSION}",
        headers=HEADERS, json=schema
    )
    if r.status_code in (200, 201):
        print(f"  Index created: {INDEX_NAME} ({len(schema['fields'])} fields)")
    else:
        raise Exception(f"Index failed: {r.status_code} {r.text}")


# ─────────────────────────────────────────────────────────────────
# STEP 2 — Create the Data Source
# Tells the indexer WHERE to find the documents
# ─────────────────────────────────────────────────────────────────
def create_datasource():
    print("\nStep 2: Creating data source...")
    datasource = {
        "name": DATASOURCE_NAME,
        "type": "azureblob",
        "credentials": {
            "connectionString": STORAGE_CONN
        },
        "container": {
            "name": CONTAINER_NAME,
            "query": ""          # empty = index all blobs in container
        },
        "dataChangeDetectionPolicy": {
            "@odata.type": "#Microsoft.Azure.Search.HighWaterMarkChangeDetectionPolicy",
            "highWaterMarkColumnName": "metadata_storage_last_modified"
        },
        "dataDeletionDetectionPolicy": {
            "@odata.type": "#Microsoft.Azure.Search.SoftDeleteColumnDeletionDetectionPolicy",
            "softDeleteColumnName":  "IsDeleted",
            "softDeleteMarkerValue": "true"
        }
    }
    r = requests.put(
        f"{SEARCH_ENDPOINT}/datasources/{DATASOURCE_NAME}?api-version={API_VERSION}",
        headers=HEADERS, json=datasource
    )
    if r.status_code in (200, 201):
        print(f"  Data source created: {DATASOURCE_NAME}")
        print(f"  Container: {CONTAINER_NAME}")
        print(f"  Change detection: HighWaterMark on last_modified")
    else:
        raise Exception(f"Datasource failed: {r.status_code} {r.text}")


# ─────────────────────────────────────────────────────────────────
# STEP 3 — Create the Skillset
# Defines the AI enrichment pipeline applied to each document
# ─────────────────────────────────────────────────────────────────
def create_skillset():
    print("\nStep 3: Creating skillset (AI enrichment pipeline)...")
    skillset = {
        "name": SKILLSET_NAME,
        "description": "Extract keyphrases, entities, and language from contract PDFs",
        "skills": [
            # Skill 1 — Language Detection
            # Input:  raw text from the PDF
            # Output: language code (en, fr, ar, etc.)
            {
                "@odata.type": "#Microsoft.Skills.Text.LanguageDetectionSkill",
                "name":        "language-detection",
                "description": "Detects the language of each document",
                "context":     "/document",
                "inputs":  [{"name": "text", "source": "/document/content"}],
                "outputs": [{"name": "languageCode", "targetName": "language"}]
            },
            # Skill 2 — Key Phrase Extraction
            # Input:  raw text + language code
            # Output: list of key phrases ["monthly fee", "termination clause", ...]
            {
                "@odata.type": "#Microsoft.Skills.Text.KeyPhraseExtractionSkill",
                "name":        "keyphrase-extraction",
                "description": "Extracts key phrases from contract text",
                "context":     "/document",
                "maxKeyPhraseCount": 20,
                "inputs": [
                    {"name": "text",         "source": "/document/content"},
                    {"name": "languageCode", "source": "/document/language"}
                ],
                "outputs": [{"name": "keyPhrases", "targetName": "keyphrases"}]
            },
            # Skill 3 — Entity Recognition
            # Input:  raw text + language code
            # Output: persons, organizations, locations extracted automatically
            {
                "@odata.type": "#Microsoft.Skills.Text.V3.EntityRecognitionSkill",
                "name":        "entity-recognition",
                "description": "Extracts persons, organizations, and locations",
                "context":     "/document",
                "categories":  ["Person", "Organization", "Location"],
                "inputs": [
                    {"name": "text",         "source": "/document/content"},
                    {"name": "languageCode", "source": "/document/language"}
                ],
                "outputs": [
                    {"name": "persons",       "targetName": "persons"},
                    {"name": "organizations", "targetName": "organizations"},
                    {"name": "locations",     "targetName": "locations"}
                ]
            }
        ]
    }
    r = requests.put(
        f"{SEARCH_ENDPOINT}/skillsets/{SKILLSET_NAME}?api-version={API_VERSION}",
        headers=HEADERS, json=skillset
    )
    if r.status_code in (200, 201):
        print(f"  Skillset created: {SKILLSET_NAME}")
        print(f"  Skills: language detection, key phrases, entity recognition")
    else:
        raise Exception(f"Skillset failed: {r.status_code} {r.text}")


# ─────────────────────────────────────────────────────────────────
# STEP 4 — Create the Indexer
# Wires datasource + skillset + index together
# Defines the field mappings and schedule
# ─────────────────────────────────────────────────────────────────
def create_indexer():
    print("\nStep 4: Creating indexer...")
    indexer = {
        "name":           INDEXER_NAME,
        "dataSourceName": DATASOURCE_NAME,
        "skillsetName":   SKILLSET_NAME,
        "targetIndexName": INDEX_NAME,
        "schedule": {
            # Run every 24 hours automatically
            # In production: hourly for high-frequency uploads
            "interval": "PT24H",
            "startTime": "2026-05-04T00:00:00Z"
        },
        "parameters": {
            "batchSize": 10,          # process 10 blobs per batch
            "maxFailedItems": 5,      # stop after 5 failures
            "maxFailedItemsPerBatch": 2,
            "configuration": {
                "dataToExtract":      "contentAndMetadata",  # text + all metadata
                "parsingMode":        "default",             # one doc per blob
                "pdfTextRotationAlgorithm": "detectAngles"  # handle rotated PDFs
            }
        },
        # Field mappings — blob metadata → index fields
        "fieldMappings": [
            {
                "sourceFieldName": "metadata_storage_path",
                "targetFieldName": "id",
                "mappingFunction": {"name": "base64Encode"}
                # base64Encode is REQUIRED — blob paths contain / and : which
                # are invalid in index key fields
            },
            {"sourceFieldName": "metadata_storage_name",          "targetFieldName": "metadata_storage_name"},
            {"sourceFieldName": "metadata_storage_size",          "targetFieldName": "metadata_storage_size"},
            {"sourceFieldName": "metadata_storage_last_modified", "targetFieldName": "metadata_storage_last_modified"},
            {"sourceFieldName": "metadata_title",                 "targetFieldName": "metadata_title"},
        ],
        # Output field mappings — skillset outputs → index fields
        "outputFieldMappings": [
            {"sourceFieldName": "/document/language",      "targetFieldName": "language"},
            {"sourceFieldName": "/document/keyphrases",    "targetFieldName": "keyphrases"},
            {"sourceFieldName": "/document/persons",       "targetFieldName": "persons"},
            {"sourceFieldName": "/document/organizations", "targetFieldName": "organizations"},
            {"sourceFieldName": "/document/locations",     "targetFieldName": "locations"},
        ]
    }
    r = requests.put(
        f"{SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}?api-version={API_VERSION}",
        headers=HEADERS, json=indexer
    )
    if r.status_code in (200, 201):
        print(f"  Indexer created: {INDEXER_NAME}")
        print(f"  Schedule: every 24 hours")
        print(f"  Batch size: 10 blobs per run")
    else:
        raise Exception(f"Indexer failed: {r.status_code} {r.text}")


# ─────────────────────────────────────────────────────────────────
# STEP 5 — Run the indexer NOW (do not wait for schedule)
# ─────────────────────────────────────────────────────────────────
def run_indexer():
    print("\nStep 5: Running indexer now...")
    r = requests.post(
        f"{SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}/run?api-version={API_VERSION}",
        headers=HEADERS
    )
    if r.status_code == 202:
        print(f"  Indexer started (202 Accepted)")
    else:
        raise Exception(f"Run failed: {r.status_code} {r.text}")


# ─────────────────────────────────────────────────────────────────
# STEP 6 — Poll indexer status until complete
# ─────────────────────────────────────────────────────────────────
def wait_for_indexer():
    print("\nStep 6: Waiting for indexer to complete...")
    for attempt in range(20):
        time.sleep(5)
        r = requests.get(
            f"{SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}/status?api-version={API_VERSION}",
            headers=HEADERS
        )
        data = r.json()
        last = data.get("lastResult", {})
        status = last.get("status", "unknown")
        docs_processed = last.get("itemsProcessed", 0)
        docs_failed    = last.get("itemsFailed", 0)
        print(f"  [{attempt+1:02d}] status={status}  processed={docs_processed}  failed={docs_failed}")

        if status == "success":
            print(f"\n  Indexer completed successfully")
            print(f"  Documents processed: {docs_processed}")
            print(f"  Documents failed:    {docs_failed}")

            # Print per-document details
            errors = last.get("errors", [])
            warnings = last.get("warnings", [])
            if errors:
                print(f"  Errors:")
                for e in errors:
                    print(f"    {e.get('key', '?')} — {e.get('errorMessage', '?')} ")
            if warnings:
                print(f"  Warnings: {len(warnings)}")
            return last

        elif status == "transientFailure":
            print(f"  Transient failure — retrying...")
        elif status in ("error", "failed"):
            errors = last.get("errors", [])
            for e in errors:
                print(f"  ERROR: {e}")
            raise Exception(f"Indexer failed: {last}")

    print("  Timeout — indexer still running")
    return {}


# ─────────────────────────────────────────────────────────────────
# STEP 7 — Search the indexed blobs
# ─────────────────────────────────────────────────────────────────
def search_indexed_blobs():
    print("\nStep 7: Searching indexed blob content...")
    time.sleep(5)  # wait for index to commit

    queries = [
        ("termination fee",         {"search": "termination fee",   "select": "metadata_storage_name,keyphrases,persons,locations", "top": "3"}),
        ("monthly fee enterprise",  {"search": "monthly fee enterprise", "select": "metadata_storage_name,keyphrases", "top": "3"}),
        ("filter by language=en",   {"search": "*", "filter": "language eq 'en'", "select": "metadata_storage_name,language", "$count": "true"}),
    ]

    for label, params in queries:
        print(f"\n  Query: {label}")
        r = requests.get(
            f"{SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs",
            headers=HEADERS,
            params={"api-version": API_VERSION, **params}
        )
        data = r.json()
        results = data.get("value", [])
        if not results:
            print(f"    No results yet")
        for doc in results:
            name = doc.get("metadata_storage_name", "?")
            kp   = doc.get("keyphrases", [])[:5]
            p    = doc.get("persons", [])
            loc  = doc.get("locations", [])
            score = doc.get("@search.score", 0)
            print(f"    [{score:.3f}] {name}")
            if kp:  print(f"      keyphrases:    {kp}")
            if p:   print(f"      persons:       {p}")
            if loc: print(f"      locations:     {loc}")


# ─────────────────────────────────────────────────────────────────
# STEP 8 — Save full indexer run summary
# ─────────────────────────────────────────────────────────────────
def save_summary(indexer_result):
    summary = {
        "indexer_name":       INDEXER_NAME,
        "datasource":         DATASOURCE_NAME,
        "skillset":           SKILLSET_NAME,
        "index":              INDEX_NAME,
        "container":          CONTAINER_NAME,
        "items_processed":    indexer_result.get("itemsProcessed", 0),
        "items_failed":       indexer_result.get("itemsFailed", 0),
        "status":             indexer_result.get("status", ""),
        "start_time":         indexer_result.get("startTime", ""),
        "end_time":           indexer_result.get("endTime", ""),
        "skills_applied":     ["language-detection", "keyphrase-extraction", "entity-recognition"],
        "schedule":           "PT24H"
    }
    path = "./outputs/week8/indexer_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {path}")
    return summary


# ─────────────────────────────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("BLOB INDEXER PIPELINE — Week 8 Thursday")
print("=" * 65)
print(f"Container:  {CONTAINER_NAME}")
print(f"Index:      {INDEX_NAME}")
print(f"Skills:     language + keyphrases + entities")

print("\nCleaning up existing resources...")
delete_if_exists("indexers",   INDEXER_NAME)
delete_if_exists("skillsets",  SKILLSET_NAME)
delete_if_exists("datasources", DATASOURCE_NAME)
delete_if_exists("indexes",    INDEX_NAME)

create_index()
create_datasource()
create_skillset()
create_indexer()
run_indexer()
indexer_result = wait_for_indexer()
search_indexed_blobs()
summary = save_summary(indexer_result)

print()
print("=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(json.dumps(summary, indent=2))
print()
print("Thursday Week 8 COMPLETE")
print()
print("WHAT THIS PIPELINE DOES AUTOMATICALLY EVERY 24 HOURS:")
print("  1. Scans training-forms container for new/modified PDFs")
print("  2. Extracts text from each PDF")
print("  3. Detects language")
print("  4. Extracts key phrases")
print("  5. Recognizes persons, organizations, locations")
print("  6. Updates the search index")
print("  Zero code runs after today — fully automated")
