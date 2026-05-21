import os
import requests

SPEECH_KEY    = "6yQKLUGc1Ffo4SUPVgxvYb4NizTaQaaRDgXdAYRsnD6OmgtYFj0iJQQJ99CEACYeBjFXJ3w3AAAYACOGbcaE"
SPEECH_REGION = "eastus"

os.makedirs("./outputs/week7/audio", exist_ok=True)

def tts(text, voice, filename, description, ssml=None):
    print(f"Generating: {description}")

    token_resp = requests.post(
        f"https://{SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
        headers={"Ocp-Apim-Subscription-Key": SPEECH_KEY}
    )
    if token_resp.status_code != 200:
        print(f"  FAILED to get token: {token_resp.status_code}")
        return
    token = token_resp.text

    if ssml is None:
        ssml = f'''<speak version="1.0"
            xmlns="http://www.w3.org/2001/10/synthesis"
            xml:lang="en-US">
            <voice name="{voice}">{text}</voice>
        </speak>'''

    response = requests.post(
        f"https://{SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3",
        },
        data=ssml.encode("utf-8")
    )

    path = f"./outputs/week7/audio/{filename}"
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"  Saved: {filename} ({os.path.getsize(path):,} bytes)")
    else:
        print(f"  FAILED: {response.status_code} {response.text}")

print("=" * 60)
print("SPEECH SYNTHESIS — Wednesday Week 7")
print("=" * 60)
print()

tts(
    "Hello! Welcome to ConnectPlus. How can I help you today?",
    "en-US-JennyNeural", "01_greeting.mp3", "Plain greeting"
)

tts(
    "Hello Ahmed. This is ConnectPlus calling. We noticed you have had difficulties recently and want to help. We are offering you a special discount on your annual plan.",
    "en-US-JennyNeural", "02_retention.mp3", "Retention call"
)

tts(None, "en-US-JennyNeural", "03_ssml.mp3", "SSML with pauses", ssml='''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
<voice name="en-US-JennyNeural">
    Hello Ahmed. <break time="400ms"/>
    This is ConnectPlus calling about your account. <break time="300ms"/>
    Your plan expires in <emphasis level="moderate">7 days.</emphasis>
    <break time="500ms"/>
    <prosody rate="slow">We have a special offer just for you.</prosody>
</voice></speak>''')

tts(None, "en-US-GuyNeural", "04_invoice.mp3", "Invoice notification", ssml='''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
<voice name="en-US-GuyNeural">
    Your ConnectPlus invoice is ready. <break time="300ms"/>
    Amount due: <say-as interpret-as="currency" language="en-US">EUR 53.82</say-as>
    <break time="200ms"/>
    Payment deadline: <say-as interpret-as="date" format="ymd" detail="1">2026-05-17</say-as>
    <break time="400ms"/>
    Thank you for being a valued customer.
</voice></speak>''')

tts(None, "en-US-JennyNeural", "05_churn.mp3", "Churn intervention", ssml='''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
<voice name="en-US-JennyNeural">
    <prosody rate="0.85">Hello Ahmed.</prosody>
    <break time="500ms"/>
    We noticed you have contacted support
    <emphasis level="strong">10 times</emphasis> this month.
    <break time="400ms"/>
    A senior advisor is ready to resolve all your issues personally.
    <break time="500ms"/>
    <prosody rate="slow" pitch="+5%">
        Would you like us to call you back within the next hour?
    </prosody>
</voice></speak>''')

print()
print("=" * 60)
audio_dir = "./outputs/week7/audio"
total = 0
for f in sorted(os.listdir(audio_dir)):
    if f.endswith(".mp3"):
        size = os.path.getsize(os.path.join(audio_dir, f))
        total += size
        status = "OK" if size > 1000 else "EMPTY"
        print(f"  {f:40s} {size:>8,} bytes  {status}")
print(f"  TOTAL: {total:,} bytes")
print()
print("Wednesday Week 7 COMPLETE")
