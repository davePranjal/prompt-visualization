import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Load your API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Checking available models for your API key...\n")

# 2. List all models
try:
    for m in genai.list_models():
        # Filter for models that support content generation (chat/text)
        if 'generateContent' in m.supported_generation_methods:
            print(f"Name: {m.name}")
            print(f"Display Name: {m.display_name}")
            print(f"Description: {m.description}")
            print("-" * 30)
except Exception as e:
    print(f"Error: {e}")