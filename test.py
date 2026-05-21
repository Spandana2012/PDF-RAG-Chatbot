import os
from google import genai

# Configure Gemini API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Set GOOGLE_API_KEY in your environment before running this script.")

genai.configure(api_key=api_key)

# Load Gemini model
model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

# Send prompt
response = model.generate_text(
    prompt="Explain recursion in simple words"
)

# Print response
print(getattr(response, 'text', response))