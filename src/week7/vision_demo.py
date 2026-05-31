import os
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# ───────────────────────────────
# AZURE CONFIG (PUT YOUR VALUES)
# ───────────────────────────────
ENDPOINT = "https://fayz-vision-service.cognitiveservices.azure.com/"
KEY = os.getenv("AZURE_VISION_KEY")

client = ImageAnalysisClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(KEY)
)

# ───────────────────────────────
# UI TITLE
# ───────────────────────────────
st.title("🧠 Azure Vision AI Demo (Supervisor Ready)")
st.write("Upload an image or paste a URL to analyze it using Azure AI")

# ───────────────────────────────
# INPUT OPTION
# ───────────────────────────────
mode = st.radio("Choose input type:", ["Upload Image", "Image URL"])

image_bytes = None
url = None

# ───────────────────────────────
# OPTION 1: UPLOAD IMAGE
# ───────────────────────────────
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image_bytes = uploaded.read()
        st.image(uploaded, caption="Uploaded Image")

# ───────────────────────────────
# OPTION 2: IMAGE URL
# ───────────────────────────────
else:
    url = st.text_input("Enter Image URL")
    if url:
        st.image(url, caption="Image URL Preview")

# ───────────────────────────────
# ANALYZE BUTTON
# ───────────────────────────────
if st.button("Analyze Image"):

    # CALL AZURE API
    if mode == "Upload Image" and image_bytes:
        result = client.analyze(
            image_data=image_bytes,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE,
            ]
        )

    elif mode == "Image URL" and url:
        result = client.analyze_from_url(
            image_url=url,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE,
            ]
        )
    else:
        st.error("Please provide an image first")
        st.stop()

    # ───────────────────────────
    # OUTPUT SECTION
    # ───────────────────────────

    st.subheader("🧠 AI Caption")
    if result.caption:
        st.success(result.caption.text)
        st.write("Confidence:", f"{result.caption.confidence:.2%}")

    st.subheader("🏷️ Tags")
    if result.tags:
        for t in result.tags.list[:10]:
            st.write(f"- {t.name} ({t.confidence:.2%})")

    st.subheader("📦 Objects")
    if result.objects:
        for obj in result.objects.list:
            st.write(obj.tags[0].name)

    st.subheader("👥 People Detected")
    if result.people:
        st.write(len(result.people.list))

    st.success("Analysis Complete 🚀")