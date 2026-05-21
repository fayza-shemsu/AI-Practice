import os
import json
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# ── YOUR CREDENTIALS ──────────────────────────────────────────────
# Replace these with your actual values from Azure Portal
# Portal → Your Vision Resource → Keys and Endpoint
ENDPOINT = "https://fayz-vision-service.cognitiveservices.azure.com/"
KEY      = "EVsspW5fNPdxmxoQPJeWl2zgF2g4HlEs1aJd4Br5M3qPJ1I1vFhDJQQJ99CEACYeBjFXJ3w3AAAFACOGynap"

# ── CREATE CLIENT ─────────────────────────────────────────────────
client = ImageAnalysisClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(KEY)
)

# ── IMAGE TO ANALYZE ──────────────────────────────────────────────
# This is a public image of a street scene — no need to upload anything
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"

# Use this real photo instead — a busy street scene
IMAGE_URL = "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800"

print("Analyzing image...")
print(f"URL: {IMAGE_URL}")
print()

# ── CALL THE API ──────────────────────────────────────────────────
result = client.analyze_from_url(
    image_url=IMAGE_URL,
    visual_features=[
        VisualFeatures.CAPTION,
        VisualFeatures.OBJECTS,
        VisualFeatures.TAGS,
        VisualFeatures.PEOPLE,
    ],
    language="en",
    gender_neutral_caption=True,
)

# ── PRINT RESULTS ─────────────────────────────────────────────────
print("=" * 60)
print("IMAGE ANALYSIS RESULTS")
print("=" * 60)

# 1. Caption — what is in the image overall
if result.caption:
    print(f"\nCaption:")
    print(f"  '{result.caption.text}'")
    print(f"  Confidence: {result.caption.confidence:.2%}")

# 2. Tags — keywords describing the image
if result.tags:
    print(f"\nTop 8 tags:")
    for tag in result.tags.list[:8]:
        bar = "█" * int(tag.confidence * 20)
        print(f"  {tag.name:20s} {tag.confidence:.2%}  {bar}")

# 3. Objects — what objects are detected and where
if result.objects:
    print(f"\nObjects detected: {len(result.objects.list)}")
    for obj in result.objects.list:
        box = obj.bounding_box
        name = obj.tags[0].name
        conf = obj.tags[0].confidence
        print(f"  {name:15s} confidence={conf:.2%}  "
              f"location=({box.x},{box.y})  "
              f"size={box.width}x{box.height}px")

# 4. People — how many people detected
if result.people:
    print(f"\nPeople detected: {len(result.people.list)}")
    for i, person in enumerate(result.people.list):
        box = person.bounding_box
        print(f"  Person {i+1}: confidence={person.confidence:.2%}  "
              f"location=({box.x},{box.y})  "
              f"size={box.width}x{box.height}px")

# ── SAVE RESULTS TO FILE ──────────────────────────────────────────
os.makedirs("./outputs/week7", exist_ok=True)

output = {
    "image_url": IMAGE_URL,
    "caption": result.caption.text if result.caption else None,
    "caption_confidence": result.caption.confidence if result.caption else None,
    "tags": [
        {"name": t.name, "confidence": round(t.confidence, 4)}
        for t in result.tags.list
    ] if result.tags else [],
    "objects": [
        {
            "name": obj.tags[0].name,
            "confidence": round(obj.tags[0].confidence, 4),
            "bounding_box": {
                "x": obj.bounding_box.x,
                "y": obj.bounding_box.y,
                "width": obj.bounding_box.width,
                "height": obj.bounding_box.height,
            }
        }
        for obj in result.objects.list
    ] if result.objects else [],
    "people_count": len(result.people.list) if result.people else 0,
}

with open("./outputs/week7/vision_results.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("=" * 60)
print("Results saved to ./outputs/week7/vision_results.json")
print("Monday Week 7 COMPLETE")