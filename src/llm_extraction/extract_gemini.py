import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_NAME = "gemini-3.5-flash-lite"

load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

jobs = pd.read_excel(BASE_DIR / "data" / "blind_data.xlsx")
prompt = (BASE_DIR / "prompts" / "skill_extraction_prompt.txt").read_text(encoding="utf-8")

print(f"Loaded {len(jobs)} job descriptions")

job = jobs.iloc[0]
job_id = job["job_id"]
job_title = job["Title"]
job_description = job["Job description (full text)"]

print(f"Processing job {job_id}: {job_title}")

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt + "\n" + str(job_description),
    config=types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=list[str],
    ),
)

skills = json.loads(response.text)
if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
    raise ValueError("Gemini returned an invalid skill list")

print(f"\nExtracted {len(skills)} skills:")
for i, skill in enumerate(skills, start=1):
    print(f"{i}. {skill}")
