"""Generate a secure API key for Uni_Vision.

Usage:
    python -m uni_vision.api.middleware.gen_key
"""

from __future__ import annotations

from uni_vision.api.middleware.auth import generate_api_key


def main() -> None:
    key = generate_api_key()
    print(f"Generated API key: {key}")
    print("Add it to your .env file:")
    print(f"  UV_API_API_KEYS={key}")


if __name__ == "__main__":
    main()
