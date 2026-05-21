import os
from dotenv import load_dotenv

load_dotenv()
import time
import json
import requests
from fpdf import FPDF

DI_ENDPOINT = "https://fayz-doc-intelligence.cognitiveservices.azure.com/"
DI_KEY = os.getenv("AZURE_FORM_RECOGNIZER_KEY")

os.makedirs("./outputs/week8", exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# STEP 1 — Create a realistic PDF invoice
# ─────────────────────────────────────────────────────────────────
def create_invoice_pdf():
    print("Creating PDF invoice...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "CONNECTPLUS B.V.", ln=True, align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, "123 Tech Street, Amsterdam, Netherlands", ln=True, align="C")
    pdf.cell(0, 6, "VAT: NL123456789B01  |  IBAN: NL91ABNA0417164300", ln=True, align="C")
    pdf.ln(8)

    # Invoice details
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "INVOICE", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(95, 6, "Invoice Number: INV-2026-00847")
    pdf.cell(0, 6, "Bill To:", ln=True)
    pdf.cell(95, 6, "Invoice Date: 2026-05-03")
    pdf.cell(0, 6, "Ahmed Al-Rashidi", ln=True)
    pdf.cell(95, 6, "Due Date: 2026-05-17")
    pdf.cell(0, 6, "456 Client Avenue, Dubai, UAE", ln=True)
    pdf.ln(6)

    # Table header
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(90, 8, "Description", border=1, fill=True)
    pdf.cell(25, 8, "Qty", border=1, fill=True, align="C")
    pdf.cell(35, 8, "Unit Price", border=1, fill=True, align="R")
    pdf.cell(35, 8, "Total", border=1, fill=True, align="R")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", size=10)
    items = [
        ("ConnectPlus Pro Plan - Monthly", "1", "EUR 49.99", "EUR 49.99"),
        ("Additional User Licenses x5",   "5", "EUR  4.99", "EUR 24.95"),
        ("SMS Notification Pack",          "1", "EUR  9.99", "EUR  9.99"),
        ("Priority Support Add-on",        "1", "EUR 14.99", "EUR 14.99"),
        ("Setup & Onboarding Fee",         "1", "EUR 99.00", "EUR 99.00"),
    ]
    for desc, qty, unit, total in items:
        pdf.cell(90, 7, desc, border=1)
        pdf.cell(25, 7, qty, border=1, align="C")
        pdf.cell(35, 7, unit, border=1, align="R")
        pdf.cell(35, 7, total, border=1, align="R")
        pdf.ln()

    # Totals
    pdf.ln(4)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(150, 6, "")
    pdf.cell(35, 6, "Subtotal:", align="R")
    pdf.cell(0, 6, "EUR 198.92", ln=True)
    pdf.cell(150, 6, "")
    pdf.cell(35, 6, "VAT (21%):", align="R")
    pdf.cell(0, 6, "EUR  41.77", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(150, 6, "")
    pdf.cell(35, 6, "TOTAL DUE:", align="R")
    pdf.cell(0, 6, "EUR 240.69", ln=True)

    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", size=9)
    pdf.cell(0, 5, "Payment terms: 14 days. Late payments incur 2% monthly interest.", ln=True)
    pdf.cell(0, 5, "Questions? billing@connectplus.com  |  +31 20 123 4567", ln=True)

    path = "./outputs/week8/invoice.pdf"
    pdf.output(path)
    print(f"  Saved: {path} ({os.path.getsize(path):,} bytes)")
    return path


# ─────────────────────────────────────────────────────────────────
# STEP 2 — Send PDF to Document Intelligence Layout model
# ─────────────────────────────────────────────────────────────────
def analyze_document(pdf_path):
    print("\nSending to Document Intelligence Layout model...")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Submit async job
    response = requests.post(
        f"{DI_ENDPOINT}documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30&features=keyValuePairs",
        headers={
            "Ocp-Apim-Subscription-Key": DI_KEY,
            "Content-Type": "application/pdf"
        },
        data=pdf_bytes
    )

    if response.status_code != 202:
        raise Exception(f"Submit failed: {response.status_code} {response.text}")

    operation_url = response.headers["Operation-Location"]
    print(f"  Job submitted. Polling for result...")

    # Poll until complete
    while True:
        time.sleep(2)
        result = requests.get(
            operation_url,
            headers={"Ocp-Apim-Subscription-Key": DI_KEY}
        ).json()

        status = result.get("status")
        print(f"  Status: {status}")

        if status == "succeeded":
            return result["analyzeResult"]
        elif status == "failed":
            raise Exception(f"Analysis failed: {result}")


# ─────────────────────────────────────────────────────────────────
# STEP 3 — Extract tables and key-value pairs from result
# ─────────────────────────────────────────────────────────────────
def extract_tables(analyze_result):
    print("\nExtracting tables...")
    tables = analyze_result.get("tables", [])
    print(f"  Tables found: {len(tables)}")

    extracted = []
    for t_idx, table in enumerate(tables):
        rows = table["rowCount"]
        cols = table["columnCount"]
        print(f"\n  Table {t_idx+1}: {rows} rows x {cols} columns")

        # Build a 2D grid from cells
        grid = [[""] * cols for _ in range(rows)]
        for cell in table["cells"]:
            r = cell["rowIndex"]
            c = cell["columnIndex"]
            grid[r][c] = cell["content"]

        # Print the table
        for r, row in enumerate(grid):
            prefix = "  HEADER " if r == 0 else f"  row {r:>2}  "
            print(prefix + " | ".join(f"{v:<28}" for v in row))

        extracted.append({
            "row_count": rows,
            "col_count": cols,
            "headers": grid[0] if grid else [],
            "rows": grid[1:] if len(grid) > 1 else [],
            "raw_grid": grid
        })

    return extracted


def extract_key_values(analyze_result):
    print("\nExtracting key-value pairs...")
    kvps = analyze_result.get("keyValuePairs", [])
    print(f"  Key-value pairs found: {len(kvps)}")

    result = {}
    for kv in kvps:
        key   = kv.get("key",   {}).get("content", "")
        value = kv.get("value", {}).get("content", "") if kv.get("value") else ""
        if key:
            result[key] = value
            print(f"  {key:<30} → {value}")
    return result


def extract_paragraphs(analyze_result):
    print("\nExtracting paragraphs and titles...")
    paragraphs = analyze_result.get("paragraphs", [])
    titles = [p["content"] for p in paragraphs if p.get("role") == "title"]
    print(f"  Paragraphs: {len(paragraphs)}  |  Titles: {len(titles)}")
    for t in titles:
        print(f"  Title: {t}")
    return paragraphs, titles


# ─────────────────────────────────────────────────────────────────
# STEP 4 — Build structured invoice summary
# ─────────────────────────────────────────────────────────────────
def build_invoice_summary(tables, key_values, titles):
    print("\nBuilding invoice summary...")

    # Find the items table (the one with the most rows)
    items_table = max(tables, key=lambda t: t["row_count"]) if tables else None

    items = []
    if items_table:
        for row in items_table["rows"]:
            if len(row) >= 4 and row[0]:
                items.append({
                    "description": row[0],
                    "qty":         row[1],
                    "unit_price":  row[2],
                    "total":       row[3],
                })

    # Extract totals from Table 2 (the totals table)
    subtotal = vat = total_due = ""
    if len(tables) >= 2:
        for row in tables[1]["raw_grid"]:
            joined = " ".join(row)
            if "Subtotal" in joined:
                # value is embedded in the cell text e.g. "Subtotal: EUR 198.92"
                import re
                m = re.search(r"EUR[\s]+(\d+[\.,]\d+)", joined)
                subtotal = "EUR " + m.group(1) if m else joined
            elif "VAT" in joined:
                m = re.search(r"EUR[\s]+(\d+[\.,]\d+)", joined)
                vat = "EUR " + m.group(1) if m else joined
            elif "TOTAL" in joined:
                m = re.search(r"EUR[\s]+(\d+[\.,]\d+)", joined)
                total_due = "EUR " + m.group(1) if m else joined

    # Extract invoice header fields from paragraphs via key_values
    # or fall back to scanning all paragraph content
    def get_field(label, kv):
        for k, v in kv.items():
            if label.lower() in k.lower():
                return v
        return ""

    summary = {
        "invoice_number": get_field("invoice number", key_values),
        "invoice_date":   get_field("invoice date",   key_values),
        "due_date":       get_field("due date",       key_values),
        "bill_to":        get_field("bill to",        key_values),
        "items":          items,
        "item_count":     len(items),
        "subtotal":       subtotal,
        "vat":            vat,
        "total_due":      total_due,
        "tables_found":   len(tables),
        "key_values_found": len(key_values),
    }

    path = "./outputs/week8/invoice_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {path}")
    return summary


# ─────────────────────────────────────────────────────────────────
# RUN THE COMPLETE PIPELINE
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("DOCUMENT INTELLIGENCE — Week 8 Monday")
print("=" * 65)

pdf_path       = create_invoice_pdf()
analyze_result = analyze_document(pdf_path)
tables         = extract_tables(analyze_result)
key_values     = extract_key_values(analyze_result)
paragraphs, titles = extract_paragraphs(analyze_result)
summary        = build_invoice_summary(tables, key_values, titles)

print()
print("=" * 65)
print("FINAL INVOICE SUMMARY JSON")
print("=" * 65)
print(json.dumps(summary, indent=2))

print()
print("=" * 65)
print("WHAT DOCUMENT INTELLIGENCE EXTRACTED:")
print(f"  Tables found:        {summary['tables_found']}")
print(f"  Key-value pairs:     {summary['key_values_found']}")
print(f"  Line items:          {summary['item_count']}")
print(f"  Invoice number:      {summary['invoice_number']}")
print(f"  Total due:           {summary['total_due']}")
print()
print("Monday Week 8 COMPLETE")
