# --- Imports ---
from openai import OpenAI
from typing import Optional
import argparse

# --- Initialize OpenAI Client ---
try:
    client = OpenAI()
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    exit()


# --- Argument Parsing ---
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the OpenAI API call.

    Returns:
        argparse.Namespace: Parsed command line arguments containing prompt, model, and max_tokens
    """
    parser = argparse.ArgumentParser(description="Call OpenAI API with custom prompt")
    parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help="The prompt to send to the API",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gpt-4o-mini",
        help="The model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "-t",
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens (default: 1000)",
    )
    return parser.parse_args()


# --- API Interaction ---
def call_api(
    prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 1000
) -> Optional[str]:
    """
    Calls the OpenAI API with the given parameters and returns the response.

    Args:
        prompt (str): The input text to send to the API
        model (str): The OpenAI model to use
        max_tokens (int): Maximum number of tokens in the response

    Returns:
        Optional[str]: The API response text if successful, None if an error occurs
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None


# --- Main Execution ---
def main() -> None:
    """
    Main function to execute the API call with command line arguments.
    Parses arguments and prints the API response.
    """
    args = parse_arguments()
    response = call_api(
        prompt=args.prompt, model=args.model, max_tokens=args.max_tokens
    )
    print(response)


# --- Script Entry Point ---
if __name__ == "__main__":
    main()
