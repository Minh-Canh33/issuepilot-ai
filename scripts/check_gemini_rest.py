
import httpx
from dotenv import dotenv_values

config = dotenv_values(".env")
api_key = config.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY")

model = config.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

url = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{model}:generateContent"
)

response = httpx.post(
    url,
    headers={
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    },
    json={
        "contents": [
            {"parts": [{"text": "Say OK"}]}
        ]
    },
    timeout=30,
)

print("HTTP status:", response.status_code)

if response.is_success:
    print("SUCCESS: Gemini REST API works")
else:
    error = response.json().get("error", {})
    print("Error:", error.get("message"))
    print("Reason:", next(
        (
            item.get("reason")
            for item in error.get("details", [])
            if "reason" in item
        ),
        "unknown",
    ))
