import os
from azure.storage.blob import BlobServiceClient

# Get this from:
# Portal → Storage accounts → your account → Shared access signature
# Check: Container + Object + Read + Write + List → Generate SAS
# Copy the "Blob service SAS URL"
STORAGE_SAS_URL    = "https://aiinterns.blob.core.windows.net/?sv=2025-11-05&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2026-05-04T05:52:48Z&st=2026-05-03T21:37:48Z&spr=https&sig=IJFRlRqLvo2s3AyqUzdo0OGjmZfcS%2BZAXdb2Wr0SW9g%3D"
CONTAINER_NAME     = "training-forms"

blob_service = BlobServiceClient(account_url=STORAGE_SAS_URL)

try:
    blob_service.create_container(CONTAINER_NAME)
    print(f"Container created: {CONTAINER_NAME}")
except Exception as e:
    print(f"Container exists or error: {e}")

container_client = blob_service.get_container_client(CONTAINER_NAME)

forms_dir = "./outputs/week8/training_forms"
for filename in sorted(os.listdir(forms_dir)):
    if filename.endswith(".pdf"):
        path = os.path.join(forms_dir, filename)
        with open(path, "rb") as f:
            container_client.upload_blob(name=filename, data=f, overwrite=True)
        print(f"  Uploaded: {filename}")

print()
print("All 5 forms uploaded to Blob Storage")
print(f"Container URL for Studio: {STORAGE_SAS_URL}/{CONTAINER_NAME}")
