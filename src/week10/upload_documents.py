
import os
from dotenv import load_dotenv

load_dotenv()
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

AZURE_OAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OAI_KEY = os.getenv("AZURE_OPENAI_KEY")
SEARCH_ENDPOINT    = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME         = "connectplus-rag"
DEPLOY_EMBED       = "text-embedding-3-small"

oai_client    = AzureOpenAI(
    azure_endpoint=AZURE_OAI_ENDPOINT,
    api_key=AZURE_OAI_KEY,
    api_version="2024-02-01"
)
search_client = SearchClient(
    SEARCH_ENDPOINT,
    INDEX_NAME,
    AzureKeyCredential(SEARCH_KEY)
)

DOCUMENTS = [
    {
        "source": "cancellation_policy.txt",
        "title":  "Cancellation Policy",
        "chunks": [
            "ConnectPlus cancellation requires 30 days written notice to cancellations@connectplus.co.uk. No fee within first 14 days cooling-off period under Consumer Rights Act 2015.",
            "After 14 days: £25 early termination fee for contracts under 12 months. After 12 months: no termination fee. After 24 months: customer free to leave anytime.",
            "Equipment return required within 14 days of cancellation. Non-return fee: £75 on final bill. Return label sent via email within 2 working days.",
            "Final bill issued within 5 working days. Pro-rata refund calculated from cancellation date. Refunds to original payment method within 7 working days.",
        ]
    },
    {
        "source": "retention_playbook.txt",
        "title":  "Retention Playbook",
        "chunks": [
            "HIGH RISK defined as two or more of: explicit cancel intent, 3+ support calls in 30 days, plan downgrade in 14 days, billing complaint this month.",
            "HIGH RISK actions: acknowledge frustration first, offer 20% loyalty discount for 3 months maximum, assign dedicated Tier-2 agent, priority engineer visit within 48 hours if technical issue.",
            "MEDIUM RISK: one signal present. Actions: one-month bill credit immediately, free router upgrade if over 2 years old, satisfaction survey at end of call.",
            "LOW RISK: general dissatisfaction. Actions: NPS survey, £20 referral bonus credit, information about loyalty rewards programme.",
            "ESCALATE to supervisor immediately if customer mentions: legal action, Ofcom, media threat, safeguarding concern, bereavement.",
            "Discount authorisation limits: Agent max 20% for 3 months. Team Leader max 30% for 6 months. Manager max 50% for 12 months requires written approval.",
        ]
    },
    {
        "source": "broadband_plans.txt",
        "title":  "Broadband Plans",
        "chunks": [
            "Essential Plan: 35Mbps download, 5Mbps upload, £25 per month, no minimum term, free router included.",
            "Standard Plan: 67Mbps download, 15Mbps upload, £35 per month, 12-month contract, includes TV app.",
            "Premium Plan: 150Mbps download, 30Mbps upload, £50 per month, 12-month contract, priority support, TV app, static IP option.",
            "Ultrafast Plan: 500Mbps download, 50Mbps upload, £70 per month, 24-month contract, dedicated support line, 2 static IPs.",
            "Speed Promise: if you receive less than 50% of advertised speed for 3 consecutive days you may exit contract without penalty. Log at speedtest.connectplus.co.uk.",
            "All plans include: no setup fee, no activation charge, free standard installation, 24/7 UK-based support.",
        ]
    },
    {
        "source": "billing_policy.txt",
        "title":  "Billing Policy",
        "chunks": [
            "Bills generated on 1st of each month for following month. Payment due within 14 days. Late payment fee £12 after grace period.",
            "Double billing error: refund within 3-5 working days. Overcharge disputes must be raised within 60 days via billing@connectplus.co.uk.",
            "Direct debit failure first time: £5 admin fee, payment retried after 5 working days. Second failure: additional £5. Third failure: account suspended, reconnection fee applies.",
            "Price increase: minimum 30 days written notice required by law. Customer may exit contract without penalty within 30 days of receiving price increase notice.",
        ]
    },
]

def embed_batch(texts):
    r = oai_client.embeddings.create(model=DEPLOY_EMBED, input=texts)
    return [d.embedding for d in r.data]

print("Embedding and uploading documents...")
print("-" * 50)

all_docs = []
for doc in DOCUMENTS:
    print(f"\nProcessing: {doc['source']}")
    embeddings = embed_batch(doc["chunks"])
    print(f"  Embedded {len(doc['chunks'])} chunks")
    for i, (text, embedding) in enumerate(zip(doc["chunks"], embeddings)):
        all_docs.append({
            "id":        f"{doc['source'].replace('.','_')}_{i}",
            "content":   text,
            "title":     doc["title"],
            "source":    doc["source"],
            "page":      i + 1,
            "embedding": embedding,
        })

print(f"\nUploading {len(all_docs)} chunks to index...")
result = search_client.upload_documents(documents=all_docs)
success = sum(1 for r in result if r.succeeded)
print(f"Uploaded: {success}/{len(all_docs)} succeeded")

from azure.search.documents.indexes import SearchIndexClient
idx_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))
stats = idx_client.get_index_statistics(INDEX_NAME)
print(f"Document count: {stats.document_count}")
print(f"Storage size:   {stats.storage_size} bytes")
print("Index is ready for RAG queries")
