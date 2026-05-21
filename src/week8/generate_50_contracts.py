import os
import random
from fpdf import FPDF

os.makedirs("./outputs/week8/fifty_contracts", exist_ok=True)

PLANS = [
    ("ConnectPlus Basic",      29.99,  12, 99.00),
    ("ConnectPlus Pro",        49.99,  24, 199.00),
    ("ConnectPlus Business",   59.99,  24, 199.00),
    ("ConnectPlus Enterprise", 79.99,  36, 299.00),
    ("ConnectPlus Ultimate",   99.99,  36, 399.00),
]

CUSTOMERS = [
    ("Ahmed Al-Rashidi",   "456 Client Ave, Dubai, UAE",          "UAE"),
    ("Sara Mensah",        "12 Palm Road, Accra, Ghana",          "Ghana"),
    ("Luca Bianchi",       "Via Roma 88, Milan, Italy",           "Italy"),
    ("Yuki Tanaka",        "3-5 Shibuya, Tokyo, Japan",           "Japan"),
    ("Fatima Al-Zahra",    "King Fahd Road, Riyadh, Saudi Arabia","Saudi Arabia"),
    ("Carlos Rivera",      "Av. Reforma 200, Mexico City, Mexico","Mexico"),
    ("Aisha Nkrumah",      "Ring Road, Kumasi, Ghana",            "Ghana"),
    ("Hiroshi Yamamoto",   "Shinjuku 1-1, Tokyo, Japan",          "Japan"),
    ("Elena Petrov",       "Tverskaya 10, Moscow, Russia",        "Russia"),
    ("David Okonkwo",      "Victoria Island, Lagos, Nigeria",     "Nigeria"),
    ("Maria Santos",       "Rua Augusta 50, Lisbon, Portugal",    "Portugal"),
    ("Wei Zhang",          "Nanjing Road, Shanghai, China",       "China"),
    ("Amara Diallo",       "Avenue Cheikh Anta Diop, Dakar, Senegal", "Senegal"),
    ("Sophie Dubois",      "Rue de Rivoli 100, Paris, France",    "France"),
    ("James Osei",         "Oxford Street, Accra, Ghana",         "Ghana"),
    ("Priya Sharma",       "MG Road, Bangalore, India",           "India"),
    ("Omar Hassan",        "Tahrir Square 5, Cairo, Egypt",       "Egypt"),
    ("Isabella Rossi",     "Corso Italia 30, Rome, Italy",        "Italy"),
    ("Kenji Watanabe",     "Ginza 4-6, Tokyo, Japan",             "Japan"),
    ("Fatou Coulibaly",    "Boulevard de la Republique, Bamako, Mali", "Mali"),
]

ADDONS = [
    "Priority Support Add-on",
    "SMS Notification Pack",
    "Extra Storage 100GB",
    "Advanced Analytics Dashboard",
    "API Access License",
    "White-label Branding",
    "Custom Domain",
    "Dedicated Account Manager",
    "SLA 99.9% Guarantee",
    "Multi-region Failover",
]

def create_contract(contract_num, customer, plan, start_date, addons):
    name, address, country = customer
    plan_name, monthly_fee, duration, termination_fee = plan
    addon_total = sum(4.99 * (i+1) for i in range(len(addons)))
    total_monthly = monthly_fee + addon_total

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "CONNECTPLUS B.V.", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, "SERVICE AGREEMENT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 5, "123 Tech Street, Amsterdam, Netherlands", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "CONTRACT DETAILS", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)

    fields = [
        ("Contract Number:",    contract_num),
        ("Start Date:",         start_date),
        ("Plan Name:",          plan_name),
        ("Contract Duration:",  f"{duration} months"),
        ("Monthly Fee:",        f"EUR {monthly_fee:.2f}"),
        ("Termination Fee:",    f"EUR {termination_fee:.2f}"),
        ("Total Monthly:",      f"EUR {total_monthly:.2f}"),
    ]
    for label, value in fields:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 7, label)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "CUSTOMER INFORMATION", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for label, value in [("Customer Name:", name), ("Address:", address), ("Country:", country)]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 7, label)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

    if addons:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "ADD-ON SERVICES", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        for i, addon in enumerate(addons):
            pdf.cell(0, 6, f"  - {addon}  (EUR {4.99*(i+1):.2f}/mo)", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "TERMS AND CONDITIONS", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    terms = [
        "1. Service activates within 3 business days of signing.",
        "2. Monthly fees billed on the 1st of each month.",
        "3. Early termination requires 30 days written notice.",
        f"4. Termination fee EUR {termination_fee:.2f} applies if cancelled before {duration} months.",
        "5. ConnectPlus reserves the right to modify terms with 30 days notice.",
        "6. Disputes resolved under Netherlands law, Amsterdam jurisdiction.",
        "7. Data processed under GDPR. See privacy policy at connectplus.com/privacy.",
        f"8. SLA: 99.5% uptime guaranteed. Credits issued for downtime exceeding threshold.",
    ]
    for term in terms:
        pdf.cell(0, 5, term, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(80, 7, "Customer Signature: ________________")
    pdf.cell(0, 7, f"Date: {start_date}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(80, 7, "ConnectPlus Representative: ________________")
    pdf.cell(0, 7, "Date: ________________", new_x="LMARGIN", new_y="NEXT")

    path = f"./outputs/week8/fifty_contracts/{contract_num}.pdf"
    pdf.output(path)
    return path

random.seed(42)
dates = [
    "2026-01-05", "2026-01-12", "2026-01-19", "2026-01-28",
    "2026-02-03", "2026-02-11", "2026-02-17", "2026-02-24",
    "2026-03-02", "2026-03-09", "2026-03-16", "2026-03-23",
    "2026-04-01", "2026-04-08", "2026-04-14", "2026-04-20",
    "2026-04-25", "2026-05-01", "2026-05-02", "2026-05-03",
]

print("Generating 50 contract PDFs...")
for i in range(50):
    contract_num = f"CNT-2026-{1000+i:04d}"
    customer     = CUSTOMERS[i % len(CUSTOMERS)]
    plan         = PLANS[i % len(PLANS)]
    start_date   = dates[i % len(dates)]
    num_addons   = random.randint(0, 3)
    addons       = random.sample(ADDONS, num_addons)
    path = create_contract(contract_num, customer, plan, start_date, addons)
    print(f"  {contract_num}  {customer[0]:<22} {plan[0]:<25} {start_date}")

print(f"\n50 PDFs created in ./outputs/week8/fifty_contracts/")
folder = "./outputs/week8/fifty_contracts"
all_files = [os.path.join(folder, x) for x in os.listdir(folder)]
print(f"Total size: {sum(os.path.getsize(x) for x in all_files):,} bytes")
