#!/usr/bin/env python3
"""Small terminal chat client for custom/self-hosted model endpoints."""

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("HOST_BASE_URL", "https://overprice-decaf-naming.ngrok-free.dev/v1")
MODEL = os.getenv("HOST_MODEL", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
ENV_FILE = Path(__file__).with_name(".env")
DEFAULT_API_KEY = "yourapikey"


def load_api_key() -> str:
    """Read HOST_API_KEY from environment or .env, falling back to DEFAULT_API_KEY."""
    key = os.getenv("HOST_API_KEY") or os.getenv("HOST_API_KEY".lower())
    if key:
        return key.strip()

    try:
        contents = ENV_FILE.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError):
        contents = ""

    for line in contents.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            if name.strip() in {"HOST_API_KEY", "API_KEY"}:
                return value.strip().strip('"').strip("'")

    return DEFAULT_API_KEY


def ask(api_key: str, messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
        }
    ).encode("utf-8")
    request = Request(
        f"{BASE_URL.rstrip('/')}/chat/completions",
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
        raise RuntimeError(f"Could not connect to host: {error.reason}") from error

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Unexpected API response: {data}") from error


def main() -> None:
    try:
        api_key = load_api_key()
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    messages: list[dict[str, str]] = []
    display_name = MODEL.split("/")[-1]

    print(f"Chatting with {MODEL}. Type 'exit' or press Ctrl-D to quit.")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": prompt})
        try:
            answer = ask(api_key, messages)
        except RuntimeError as error:
            messages.pop()
            print(f"Error: {error}", file=sys.stderr)
            continue

        messages.append({"role": "assistant", "content": answer})
        print(f"{display_name}: {answer}\n")


if __name__ == "__main__":
    main()