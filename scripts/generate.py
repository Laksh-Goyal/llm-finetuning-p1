import requests
import argparse

API_URL = "http://localhost:8000/generate"

def generate(prompt: str, max_tokens: int = 256):
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Request failed: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input text prompt")
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    out = generate(args.prompt, args.max_tokens)
    print(">> Generated Response:")
    print(out)
