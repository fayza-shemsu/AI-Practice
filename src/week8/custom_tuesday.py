import os
import json
import time
import requests
from fpdf import FPDF

os.makedirs("./outputs/week8/training_forms", exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# CREATE 5 VARIATIONS OF THE SAME FORM
# Each has the same fields but slightly different values
# This is what you label in the Studio UI
# ─────────────────────────────────────────────────────────────────
contracts = [
    {
        "contract_number":    "CNT-2026-0847",
        "customer_name":      "Ahmed Al-Rashidi",
        "customer_address":   "456 Client Avenue, Dubai, UAE",
        "monthly_fee":        "EUR 49.99",
        "contract_duration":  "24 months",
        "termination_fee":    "EUR 199.00",
        "start_date":         "2026-05-01",
        "plan_name":          "ConnectPlus Pro",
    },
    {
        "contract_number":    "CNT-2026-0901",
        "customer_name":      "Sara Mensah",
        "customer_address":   "12 Palm Road, Accra, Ghana",
        "monthly_fee":        "EUR 29.99",
        "contract_duration":  "12 months",
        "termination_fee":    "EUR 99.00",
        "start_date":         "2026-05-03",
        "plan_name":          "ConnectPlus Basic",
    },
    {
        "contract_number":    "CNT-2026-0955",
        "customer_name":      "Luca Bianchi",
        "customer_address":   "Via Roma 88, Milan, Italy",
        "monthly_fee":        "EUR 79.99",
        "contract_duration":  "36 months",
        "termination_fee":    "EUR 299.00",
        "start_date":         "2026-04-15",
        "plan_name":          "ConnectPlus Enterprise",
    },
    {
        "contract_number":    "CNT-2026-1002",
        "customer_name":      "Yuki Tanaka",
        "customer_address":   "3-5 Shibuya, Tokyo, Japan",
        "monthly_fee":        "EUR 49.99",
        "contract_duration":  "24 months",
        "termination_fee":    "EUR 199.00",
        "start_date":         "2026-04-20",
        "plan_name":          "ConnectPlus Pro",
    },
    {
        "contract_number":    "CNT-2026-1078",
        "customer_name":      "Fatima Al-Zahra",
        "customer_address":   "King Fahd Road, Riyadh, KSA",
        "monthly_fee":        "EUR 59.99",
        "contract_duration":  "24 months",
        "termination_fee":    "EUR 199.00",
        "start_date":         "2026-05-01",
        "plan_name":          "ConnectPlus Business",
    },
]

def create_contract_pdf(data, filename):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "CONNECTPLUS B.V.", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, "SERVICE AGREEMENT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, "123 Tech Street, Amsterdam, Netherlands", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Contract fields — fixed layout (same position every time)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "CONTRACT DETAILS", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.ln(2)

    fields = [
        ("Contract Number:",   data["contract_number"]),
        ("Start Date:",        data["start_date"]),
        ("Plan Name:",         data["plan_name"]),
        ("Contract Duration:", data["contract_duration"]),
        ("Monthly Fee:",       data["monthly_fee"]),
        ("Termination Fee:",   data["termination_fee"]),
    ]

    for label, value in fields:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 8, label)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 8, value, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "CUSTOMER INFORMATION", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 8, "Customer Name:")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, data["customer_name"], new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 8, "Customer Address:")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, data["customer_address"], new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)

    # Terms
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "TERMS AND CONDITIONS", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    terms = [
        "1. Service will be activated within 3 business days of contract signing.",
        "2. Monthly fees are billed on the 1st of each month.",
        "3. Early termination requires written notice 30 days in advance.",
        f"4. Termination fee applies if contract is cancelled before {data['contract_duration']} term.",
        "5. ConnectPlus B.V. reserves the right to modify service terms with 30 days notice.",
    ]
    for term in terms:
        pdf.cell(0, 6, term, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(80, 8, "Customer Signature: ________________")
    pdf.cell(0, 8, "Date: ________________", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.cell(80, 8, "ConnectPlus Representative: ________________")
    pdf.cell(0, 8, "Date: ________________", new_x="LMARGIN", new_y="NEXT")

    path = f"./outputs/week8/training_forms/{filename}"
    pdf.output(path)
    return path

print("Creating 5 training contract forms...")
for i, contract in enumerate(contracts, 1):
    path = create_contract_pdf(contract, f"contract_{i:02d}.pdf")
    size = os.path.getsize(path)
    print(f"  contract_{i:02d}.pdf  ({size:,} bytes)  — {contract['customer_name']}")

print()
print("5 training forms created in ./outputs/week8/training_forms/")
print()
print("NEXT STEP — Upload to Azure Blob Storage:")
print("  You need a Storage Account SAS URL to upload these.")
print("  Portal → Storage accounts → your account → Shared access signature")
print("  → check: Container, Object, Read, Write, List → Generate SAS")
print("  Copy the Blob service SAS URL and paste it when prompted.")
