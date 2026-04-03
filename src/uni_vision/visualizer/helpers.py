"""Shared helpers for visualizer modules."""

from __future__ import annotations

from typing import Any

import streamlit as st


def get_api_base() -> str:
    """Return the API base URL from sidebar or default."""
    return st.sidebar.text_input("API Base URL", value="http://localhost:8000")


def get_api_key() -> str:
    """Return the API key from sidebar input."""
    return st.sidebar.text_input("API Key", type="password", value="")


def api_headers(api_key: str = "") -> dict[str, str]:
    """Build request headers with optional API key."""
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def fetch_json(url: str, api_key: str = "") -> dict[str, Any] | None:
    """Fetch JSON from the API, returning None on errors."""
    import httpx

    try:
        resp = httpx.get(url, headers=api_headers(api_key), timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        st.error(f"API request failed: {exc}")
        return None
