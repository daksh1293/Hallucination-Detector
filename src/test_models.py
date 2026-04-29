from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Most likely active models on Groq right now
test_models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-70b-versatile",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "llama-3.1-70b-specdec",
    "llama-3.2-1b-preview",
    "mistral-7b-instruct",
    "gemma-7b-it",
]

print("Testing available models...\n")
for model in test_models:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5
        )
        print(f"✅ WORKING: {model}")
    except Exception as e:
        print(f"❌ FAILED:  {model}")