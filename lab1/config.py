import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = os.environ.get("MODEL_ID")

assert HF_TOKEN is not None and len(HF_TOKEN) > 0, "HF_TOKEN is missing in .env."
assert MODEL_ID is not None and len(MODEL_ID) > 0, "MODEL_ID is missing in .env."
