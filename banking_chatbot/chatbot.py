"""
Module 3 — Banking Chatbot (lab stub)
Extended in Module 6 Project 3 with PCI DSS compliance guardrails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm_client import get_llm_client, resolve_model

load_dotenv()

logger = logging.getLogger(__name__)

FINANCIAL_KEYWORDS = frozenset({
    "balance", "transfer", "account", "statement", "loan",
    "credit", "debit", "transaction", "payment",
})

MAX_FINANCIAL_QUERIES_PER_SESSION = 20
AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "audit_log" / "chatbot_tool_calls.jsonl"

REFUSAL_RATE_LIMITED = "You have reached the financial query limit for this session. Please start a new session or contact your branch."
REFUSAL_ERROR = "We are unable to process your request right now. Please try again later."

# Candidate card numbers: 13-19 digits, optionally separated by spaces or dashes.
_PAN_PATTERN = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_session_query_counts: dict[str, int] = {}
_session_lock = threading.Lock()


def is_financial_query(user_message: str) -> bool:
    """Return True if the message looks like a financial query."""
    tokens = set(user_message.lower().split())
    return bool(tokens & FINANCIAL_KEYWORDS)


def call_claude(user_message: str, tools: list[dict[str, Any]] | None = None) -> str:
    """Send a user message to Claude and return the text response."""
    client = get_llm_client()
    response = client.messages.create(
        model=resolve_model("claude-sonnet-4-5"),
        max_tokens=4096,
        tools=tools or [],
        messages=[{"role": "user", "content": user_message}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def handle_message(user_message: str) -> str:
    """Basic chatbot handler — guardrails added in Module 6 Project 3."""
    return call_claude(user_message)


def _luhn_valid(digits: str) -> bool:
    """Return True if the digit string passes the Luhn checksum."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _mask_pan(match: re.Match[str]) -> str:
    """Mask a Luhn-valid card number to its last 4 digits; leave other digit runs alone."""
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    if not (13 <= len(digits) <= 19) or not _luhn_valid(digits):
        return raw
    return "*" * (len(digits) - 4) + digits[-4:]


def _mask_pii(text: str) -> str:
    """Mask card numbers, SSNs, and email addresses in free text."""
    text = _PAN_PATTERN.sub(_mask_pan, text)
    text = _SSN_PATTERN.sub(lambda m: "***-**-" + m.group(0)[-4:], text)
    text = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    return text


def pre_response_pii_filter(response_text: str) -> str:
    """Mask PII in a model response before it leaves the guardrail layer."""
    return _mask_pii(response_text)


def rate_limit_check(session_id: str, user_message: str) -> bool:
    """Count financial queries per session. Return False once the session hits the limit.

    Non-financial messages are not counted and always pass.
    """
    if not is_financial_query(user_message):
        return True
    with _session_lock:
        count = _session_query_counts.get(session_id, 0)
        if count >= MAX_FINANCIAL_QUERIES_PER_SESSION:
            return False
        _session_query_counts[session_id] = count + 1
        return True


def write_audit_log_entry(
    session_id: str,
    status: str,
    user_message: str,
    is_financial: bool,
    response_length: int = 0,
) -> None:
    """Append one JSONL audit record. Only masked or hashed message content is written."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "event": "chat_turn",
        "status": status,
        "is_financial": is_financial,
        "message_sha256": hashlib.sha256(user_message.encode("utf-8")).hexdigest(),
        "message_masked": _mask_pii(user_message),
        "response_length": response_length,
    }
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_with_guardrails(session_id: str, user_message: str) -> str:
    """Rate limit, call the model, mask PII in the response, and audit the turn."""
    is_financial = is_financial_query(user_message)

    if not rate_limit_check(session_id, user_message):
        write_audit_log_entry(session_id, "rate_limited", user_message, is_financial)
        return REFUSAL_RATE_LIMITED

    try:
        raw_response = call_claude(user_message)
    except Exception as exc:
        # Log only the exception type; messages and tracebacks can echo request content.
        logger.error("chatbot model call failed: %s", type(exc).__name__)
        write_audit_log_entry(session_id, "error", user_message, is_financial)
        return REFUSAL_ERROR

    response = pre_response_pii_filter(raw_response)
    write_audit_log_entry(
        session_id, "ok", user_message, is_financial, response_length=len(response)
    )
    return response


if __name__ == "__main__":
    print(run_with_guardrails("local-demo", "What services does Heritage National Bank offer?"))
