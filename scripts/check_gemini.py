import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing")

print("Testing Gemini model:", model)

try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: API_WORKS",
    )

    print("API response:", response.text)
    print("SUCCESS: Gemini API is working!")

except Exception as error:
    print("FAILED:", type(error).__name__)
    print(str(error)[:700])