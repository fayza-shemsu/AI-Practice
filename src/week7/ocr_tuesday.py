import os
from dotenv import load_dotenv

load_dotenv()
import json
import time
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Your Vision resource credentials
ENDPOINT = "https://fayz-vision-service.cognitiveservices.azure.com/"
KEY = os.getenv("AZURE_VISION_KEY")

client = ImageAnalysisClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(KEY)
)

# ── READ TEXT FROM LOCAL IMAGE ─────────────────────────────────────
print("Reading text from receipt image...")
print()

with open("./data/week7/receipt.png", "rb") as f:
    image_data = f.read()

result = client.analyze(
    image_data=image_data,
    visual_features=[VisualFeatures.READ],
)

# ── PROCESS THE RESULTS ────────────────────────────────────────────
print("=" * 60)
print("OCR RESULTS — Full Text Extraction")
print("=" * 60)

if result.read is None:
    print("No text found")
    exit()

# Level 1 — print all text as plain string
print()
print("LEVEL 1 — PLAIN TEXT (all words joined):")
print("-" * 60)
all_text = []
for block in result.read.blocks:
    for line in block.lines:
        all_text.append(line.text)
        print(line.text)

# Level 2 — structured extraction with coordinates
print()
print("LEVEL 2 — STRUCTURED WITH COORDINATES:")
print("-" * 60)
for block_idx, block in enumerate(result.read.blocks):
    print(f"Block {block_idx + 1}:")
    for line in block.lines:
        # bounding_polygon gives 4 corner points of the text
        pts = line.bounding_polygon
        top_left = pts[0]
        print(f"  Line: '{line.text}'")
        print(f"  Position: x={top_left.x}, y={top_left.y}")
        print(f"  Words: {len(line.words)}")
        for word in line.words:
            wp = word.bounding_polygon[0]
            print(f"    '{word.text}' "
                  f"confidence={word.confidence:.2%} "
                  f"at ({wp.x},{wp.y})")
    print()

# Level 3 — business intelligence extraction
print()
print("LEVEL 3 — BUSINESS DATA EXTRACTION:")
print("-" * 60)

# Extract specific business fields from the receipt
extracted = {
    "customer_name": None,
    "account_number": None,
    "invoice_date": None,
    "total_amount": None,
    "payment_due_date": None,
    "iban": None,
    "line_items": []
}

import re

for block in result.read.blocks:
    for line in block.lines:
        text = line.text

        # Customer name
        if "Customer:" in text:
            extracted["customer_name"] = text.replace("Customer:", "").strip()

        # Account number
        if "Account:" in text:
            extracted["account_number"] = text.replace("Account:", "").strip()

        # Date
        if "Date:" in text and "due" not in text.lower():
            extracted["invoice_date"] = text.replace("Date:", "").strip()

        # Total
        if "TOTAL DUE:" in text:
            amounts = re.findall(r"EUR\s*([\d.]+)", text)
            if amounts:
                extracted["total_amount"] = "EUR " + amounts[-1]

        # Payment due date
        if "Payment due:" in text:
            extracted["payment_due_date"] = text.replace("Payment due:", "").strip()

        # IBAN
        if "IBAN:" in text:
            extracted["iban"] = text.replace("IBAN:", "").strip()

        # Line items (lines with EUR amounts but not totals)
        if "EUR" in text and "TOTAL" not in text and "VAT" not in text and "SUB" not in text:
            amounts = re.findall(r"EUR\s*([\d.]+)", text)
            if amounts:
                item_name = text.split("EUR")[0].strip()
                extracted["line_items"].append({
                    "description": item_name,
                    "amount": "EUR " + amounts[0]
                })

print("Extracted business fields:")
for key, value in extracted.items():
    if key != "line_items":
        print(f"  {key:20s}: {value}")

print()
print("Line items:")
for item in extracted["line_items"]:
    print(f"  {item['description']:30s} {item['amount']}")

# ── SAVE RESULTS ──────────────────────────────────────────────────
os.makedirs("./outputs/week7", exist_ok=True)

output = {
    "plain_text": all_text,
    "extracted_fields": extracted,
    "total_words_found": sum(
        len(line.words)
        for block in result.read.blocks
        for line in block.lines
    ),
    "total_lines_found": sum(
        len(block.lines)
        for block in result.read.blocks
    ),
}

with open("./outputs/week7/ocr_results.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("=" * 60)
print(f"Total words extracted: {output['total_words_found']}")
print(f"Total lines extracted: {output['total_lines_found']}")
print("Saved to ./outputs/week7/ocr_results.json")
print()
print("Tuesday Week 7 COMPLETE")
