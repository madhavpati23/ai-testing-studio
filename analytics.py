"""Lightweight analytics wrapper — PostHog server-side events.

Set POSTHOG_API_KEY in Streamlit Secrets (or env) to enable.
If the key is absent or PostHog is unreachable, all calls are silent no-ops.

Usage:
    import analytics
    analytics.event("certify_run", {"domain": "medical", "verdict": "PASS"})
"""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Any

import streamlit as st

_client = None
_distinct_id: str | None = None


def _get_key() -> str | None:
    try:
        val = st.secrets.get("POSTHOG_API_KEY")
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get("POSTHOG_API_KEY")


def _client_instance():
    global _client
    if _client is not None:
        return _client
    key = _get_key()
    if not key:
        return None
    try:
        from posthog import Posthog
        _client = Posthog(project_api_key=key, host="https://us.i.posthog.com")
        _client.disabled = False
    except Exception:
        _client = None
    return _client


def _session_id() -> str:
    """Stable anonymous ID for this browser session."""
    global _distinct_id
    if _distinct_id:
        return _distinct_id
    # Reuse across reruns within the same Streamlit session
    sid = st.session_state.get("_analytics_id")
    if not sid:
        sid = str(uuid.uuid4())
        st.session_state["_analytics_id"] = sid
    _distinct_id = sid
    return sid


def event(name: str, props: dict[str, Any] | None = None) -> None:
    """Fire a PostHog event. Silent no-op if key not configured."""
    try:
        ph = _client_instance()
        if ph is None:
            return
        ph.capture(distinct_id=_session_id(), event=name, properties=props or {})
    except Exception:
        pass  # never break the app for analytics


def page_view(page: str) -> None:
    event("$pageview", {"page": page})
