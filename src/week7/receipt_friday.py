import os
import re
import json
import time
import requests
from PIL import Image, ImageDraw

VISION_ENDPOINT = "https://fayz-vision-service.cognitiveservices.azure.com/"
VISION_KEY      = "EVsspW5fNPdxmxoQPJeWl2zgF2g4HlEs1aJd4Br5M3qPJ1I1vFhDJQQJ99CEACYeBjFXJ3w3AAAFACOGynap"

os.makedirs("./outputs/week7/friday", exist_ok=True)

def create_receipt_image():
    print("Creating receipt image...")
    img = Image.new("RGB", (400, 600), color="white")
    draw = ImageDraw.Draw(img)
    lines = [
        ("CONNECTPLUS CAFE", 30, "center"),
        ("123 Tech Street, Amsterdam", 60, "center"),
        ("Tel: +31 20 123 4567", 85, "center"),
        ("Date: 2026-05-03  Time: 14:32", 110, "left"),
        ("Receipt #: 00847", 130, "left"),
        ("--------------------------------", 150, "left"),
        ("ITEM                     PRICE", 165, "left"),
        ("--------------------------------", 180, "left"),
        ("Cappuccino                  4.50", 195, "left"),
        ("Club Sandwich               8.90", 215, "left"),
        ("Orange Juice                3.20", 235, "left"),
        ("Chocolate Cake              5.50", 255, "left"),
        ("Sparkling Water             2.80", 275, "left"),
        ("--------------------------------", 295, "left"),
        ("Subtotal:                  24.90", 310, "left"),
        ("VAT (21%):                  5.22", 330, "left"),
        ("--------------------------------", 348, "left"),
        ("TOTAL:                     30.12", 363, "left"),
        ("Payment: Credit Card", 385, "left"),
        ("Card: **** **** **** 4291", 405, "left"),
        ("Thank you for your visit!", 435, "center"),
    ]
    for text, y, align in lines:
        if align == "center":
            bbox = draw.textbbox((0, 0), text)
            x = (400 - (bbox[2] - bbox[0])) // 2
        else:
            x = 20
        draw.text((x, y), text, fill="black")
    path = "./outputs/week7/friday/receipt.jpg"
    img.save(path, "JPEG")
    print(f"  Receipt image saved: {path}")
    return path


def ocr_receipt(image_path):
    print("\nRunning OCR on receipt...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    response = requests.post(
        f"{VISION_ENDPOINT}vision/v3.2/read/analyze",
        headers={
            "Ocp-Apim-Subscription-Key": VISION_KEY,
            "Content-Type": "application/octet-stream"
        },
        data=image_bytes
    )
    if response.status_code != 202:
        raise Exception(f"OCR failed: {response.status_code} {response.text}")
    operation_url = response.headers["Operation-Location"]
    while True:
        time.sleep(1)
        result = requests.get(
            operation_url,
            headers={"Ocp-Apim-Subscription-Key": VISION_KEY}
        ).json()
        if result["status"] == "succeeded":
            break
        elif result["status"] == "failed":
            raise Exception("OCR failed")
    lines = []
    for page in result["analyzeResult"]["readResults"]:
        for line in page["lines"]:
            lines.append(line["text"])
    print(f"  Extracted {len(lines)} lines of text")
    print("  Raw lines:")
    for i, l in enumerate(lines):
        print(f"    [{i:02d}] {repr(l)}")
    return lines


def parse_receipt(lines):
    print("\nParsing receipt structure...")
    receipt = {
        "store_name": None,
        "date": None,
        "receipt_number": None,
        "items": [],
        "subtotal": None,
        "vat": None,
        "total": None,
        "payment_method": None,
    }

    def is_price(s):
        return bool(re.match(r"^\d+\.\d{2}$", s.strip()))

    SKIP_WORDS = {"ITEM", "PRICE", "TOTAL", "TOTAL:", "SUBTOTAL", "SUBTOTAL:",
                  "VAT", "PAYMENT", "CARD", "THANK", "PLEASE", "TEL",
                  "DATE", "RECEIPT", "STREET", "CAFE", "TIME"}

    def is_item_name(s):
        s = s.strip()
        if not s or s.startswith("-"):
            return False
        if s.upper() in SKIP_WORDS:
            return False
        for word in SKIP_WORDS:
            if s.upper().startswith(word):
                return False
        if re.match(r"^[\d\s\.,\+\*]+$", s):
            return False
        return True

    receipt["store_name"] = lines[0] if lines else None
    print(f"  Store: {receipt['store_name']}")

    for i, line in enumerate(lines):
        line = line.strip()

        # Date
        m = re.search(r"Date[:\s]+(\d{4}-\d{2}-\d{2})", line)
        if m:
            receipt["date"] = m.group(1)
            print(f"  Date: {receipt['date']}")

        # Receipt number
        m = re.search(r"Receipt\s*#[:\s]+(\w+)", line, re.IGNORECASE)
        if m:
            receipt["receipt_number"] = m.group(1)
            print(f"  Receipt#: {receipt['receipt_number']}")

        # Payment
        if re.search(r"^payment", line, re.IGNORECASE):
            receipt["payment_method"] = re.sub(r"(?i)payment:\s*", "", line).strip()
            print(f"  Payment: {receipt['payment_method']}")

        # Subtotal, VAT, Total — value on SAME line
        # Check same line first, then next line
        if re.search(r"subtotal", line, re.IGNORECASE):
            m = re.search(r"(\d+\.\d{2})", line)
            if m:
                receipt["subtotal"] = float(m.group(1))
            elif i + 1 < len(lines) and is_price(lines[i+1]):
                receipt["subtotal"] = float(lines[i+1].strip())
            if receipt["subtotal"]:
                print(f"  Subtotal: {receipt['subtotal']}")

        if re.search(r"^vat", line, re.IGNORECASE):
            m = re.search(r"(\d+\.\d{2})", line)
            if m:
                receipt["vat"] = float(m.group(1))
            elif i + 1 < len(lines) and is_price(lines[i+1]):
                receipt["vat"] = float(lines[i+1].strip())
            if receipt["vat"]:
                print(f"  VAT: {receipt['vat']}")

        if re.search(r"^total", line, re.IGNORECASE):
            m = re.search(r"(\d+\.\d{2})", line)
            if m:
                receipt["total"] = float(m.group(1))
            elif i + 1 < len(lines) and is_price(lines[i+1]):
                receipt["total"] = float(lines[i+1].strip())
            if receipt["total"]:
                print(f"  Total: {receipt['total']}")

        # Items — name + price on SAME line
        m = re.search(r"^([a-zA-Z][a-zA-Z\s]{2,})\s+(\d+\.\d{2})$", line)
        if m:
            name = m.group(1).strip()
            price = float(m.group(2))
            if is_item_name(name):
                receipt["items"].append({"name": name, "price": price})
                print(f"  Item: {name} — {price}")

        # Items — name on this line, price on NEXT line (split by OCR)
        elif is_item_name(line) and i + 1 < len(lines) and is_price(lines[i+1]):
            price = float(lines[i+1].strip())
            receipt["items"].append({"name": line, "price": price})
            print(f"  Item (split): {line} — {price}")

    return receipt


def save_summary(receipt):
    summary = {
        "store":          receipt["store_name"],
        "date":           receipt["date"],
        "receipt_number": receipt["receipt_number"],
        "items":          receipt["items"],
        "item_count":     len(receipt["items"]),
        "subtotal":       receipt["subtotal"],
        "vat":            receipt["vat"],
        "total":          receipt["total"],
        "payment_method": receipt["payment_method"],
        "currency":       "EUR",
    }
    path = "./outputs/week7/friday/receipt_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary saved: {path}")
    return summary


print("=" * 60)
print("RECEIPT SCANNER — Friday Week 7")
print("=" * 60)

receipt_path = create_receipt_image()
lines        = ocr_receipt(receipt_path)
receipt      = parse_receipt(lines)
summary      = save_summary(receipt)

print()
print("=" * 60)
print("FINAL JSON SUMMARY")
print("=" * 60)
print(json.dumps(summary, indent=2))
print()
print("=" * 60)
print("BUSINESS VALUE:")
print(f"  Items detected:    {summary['item_count']}")
print(f"  Total extracted:   EUR {summary['total']}")
print(f"  VAT extracted:     EUR {summary['vat']}")
print("  Zero manual entry  — fully automated")
print()
print("Friday Week 7 COMPLETE")
