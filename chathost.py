#!/usr/bin/env python3
"""Terminal chat client for custom/self-hosted model endpoints (e.g. vLLM via ngrok)."""

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_env_file(filepath: Path | None = None) -> None:
    """Load environment variables from a .env file if present."""
    if filepath is None:
        filepath = Path(__file__).resolve().parent / ".env"

    if not filepath.is_file():
        return

    try:
        content = filepath.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Only set if not already set in environment
            if key and key not in os.environ:
                os.environ[key] = val


# Load .env configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_env_file()

# Configuration variables
BASE_URL = os.getenv("HOST_BASE_URL", "https://your-ngrok-subdomain.ngrok-free.dev/v1").rstrip("/")
MODEL = os.getenv("HOST_MODEL", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
API_KEY = (
    os.getenv("HOST_API_KEY")
    or os.getenv("DEFAULT_API_KEY")
    or os.getenv("API_KEY")
    or "EMPTY"
).strip()


def get_chat_endpoint(base_url: str) -> str:
    """Ensure the endpoint path points to /chat/completions correctly."""
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def ask(api_key: str, messages: list[dict[str, str]], base_url: str = BASE_URL, model: str = MODEL) -> str:
    """Send a chat completion request to the OpenAI-compatible endpoint."""
    endpoint = get_chat_endpoint(base_url)
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
        }
    ).encode("utf-8")

    request = Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "chathost-python-client/1.0",
            "ngrok-skip-browser-warning": "true",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Host returned HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to host at '{endpoint}': {error.reason}") from error

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Unexpected API response format: {data}") from error


def main() -> None:
    display_name = MODEL.split("/")[-1]
    endpoint = get_chat_endpoint(BASE_URL)

    print("=" * 60)
    print(f"Model   : {MODEL}")
    print(f"Endpoint: {endpoint}")
    print("Type 'exit' or 'quit' (or Ctrl-C / Ctrl-D) to exit.")
    print("=" * 60)

    messages: list[dict[str, str]] = []

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": prompt})
        try:
            print(f"\n{display_name} is thinking...", end="\r", flush=True)
            answer = ask(API_KEY, messages)
            print(" " * (len(display_name) + 20), end="\r")  # clear thinking line
        except RuntimeError as error:
            messages.pop()
            print(f"\n[Error] {error}", file=sys.stderr)
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"{display_name}:\n{answer}")


if __name__ == "__main__":
    main()