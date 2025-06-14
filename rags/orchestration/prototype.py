# --- Imports ---
import numpy as np
from openai import OpenAI
from typing import Optional, Any

# --- 0. Initialize OpenAI Client ---
# The API key is read automatically from the OPENAI_API_KEY environment variable
try:
    client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    exit()


# --- 1. Embedding Function ---
def get_embedding(
    text: str, model: str = "text-embedding-3-small"
) -> Optional[np.ndarray[Any, Any]]:
    """Generates an embedding for the given text using OpenAI API."""
    try:
        text = text.replace("\n", " ")  # API recommendation
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error getting embedding for '{text[:30]}...': {e}")
        return None
