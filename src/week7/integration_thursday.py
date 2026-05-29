import os
from dotenv import load_dotenv

load_dotenv()
import time
import requests

VISION_ENDPOINT = "https://fayz-vision-service.cognitiveservices.azure.com/"
VISION_KEY = os.getenv("AZURE_VISION_KEY")
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION   = "eastus"

os.makedirs("./outputs/week7/thursday", exist_ok=True)

def extract_text_from_local_image(image_path):
    print(f"STEP 1 — OCR: Reading text from local image...")
    print(f"  File: {image_path}")

    submit_url = f"{VISION_ENDPOINT}vision/v3.2/read/analyze"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = requests.post(
        submit_url,
        headers={
            "Ocp-Apim-Subscription-Key": VISION_KEY,
            "Content-Type": "application/octet-stream"
        },
        data=image_bytes
    )

    if response.status_code != 202:
        print(f"  ERROR: {response.status_code} {response.text}")
        return None

    operation_url = response.headers["Operation-Location"]
    print(f"  Submitted. Waiting for result...")

    while True:
        time.sleep(1)
        result = requests.get(
            operation_url,
            headers={"Ocp-Apim-Subscription-Key": VISION_KEY}
        ).json()

        if result["status"] == "succeeded":
            break
        elif result["status"] == "failed":
            print("  OCR FAILED")
            return None
        print(f"  Status: {result['status']}...")

    lines = []
    for page in result["analyzeResult"]["readResults"]:
        for line in page["lines"]:
            lines.append(line["text"])
            print(f"  Found: '{line['text']}'")

    full_text = ". ".join(lines)
    print(f"  Full text: {full_text}")
    return full_text, lines


def speak_text(text, filename):
    print(f"\nSTEP 2 — TTS: Converting to speech...")

    token = requests.post(
        f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
        headers={"Ocp-Apim-Subscription-Key": SPEECH_KEY}
    ).text

    ssml = f"""<speak version="1.0"
        xmlns="http://www.w3.org/2001/10/synthesis"
        xml:lang="en-US">
        <voice name="en-US-JennyNeural">
            <prosody rate="0.85">{text}</prosody>
        </voice>
    </speak>"""

    response = requests.post(
        f"https://{SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
        },
        data=ssml.encode("utf-8")
    )

    path = f"./outputs/week7/thursday/{filename}"
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"  Saved: {filename} ({os.path.getsize(path):,} bytes)")
        return path
    else:
        print(f"  FAILED: {response.status_code}")
        return None


def image_to_speech(image_path, output_filename, description):
    print("=" * 60)
    print(f"PIPELINE: {description}")
    print("=" * 60)

    result = extract_text_from_local_image(image_path)
    if result is None:
        print("Pipeline failed at OCR step")
        return

    full_text, lines = result

    with open(f"./outputs/week7/thursday/{output_filename.replace('.mp3', '.txt')}", "w") as f:
        f.write(f"Image: {image_path}\n\nLines:\n")
        for line in lines:
            f.write(f"  - {line}\n")
        f.write(f"\nFull text: {full_text}\n")

    audio_path = speak_text(full_text, output_filename)

    if audio_path:
        print(f"\nPIPELINE COMPLETE")
        print(f"  Image  → {image_path}")
        print(f"  Text   → {full_text[:60]}")
        print(f"  Audio  → {audio_path}")


# Run the pipeline on our locally created image
image_to_speech(
    "./outputs/week7/thursday/test_sign.jpg",
    "sign_reading.mp3",
    "Reading a danger sign out loud"
)

print()
print("=" * 60)
print("ALL FILES:")
for f in sorted(os.listdir("./outputs/week7/thursday")):
    path = f"./outputs/week7/thursday/{f}"
    size = os.path.getsize(path)
    print(f"  {f:40s} {size:>8,} bytes")
print()
print("Thursday Week 7 COMPLETE")
