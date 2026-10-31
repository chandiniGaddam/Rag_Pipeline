"""Ticket creation through a webhook.

TICKET_WEBHOOK_URL    where the ticket JSON is POSTed (default: the built-in receiver in this app).
                      Point it at Jira / ServiceNow / Zapier / Slack-compatible endpoints to go live.
TICKET_WEBHOOK_SECRET sent as the X-Webhook-Secret header; the built-in receiver rejects requests without it.
"""
import json
import os
import secrets
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

TICKETS_FILE = Path(__file__).resolve().parent / "tickets.json"
WEBHOOK_URL = os.environ.get("TICKET_WEBHOOK_URL", "http://127.0.0.1:8000/webhook/tickets")
WEBHOOK_SECRET = os.environ.get("TICKET_WEBHOOK_SECRET") or secrets.token_urlsafe(24)

_lock = threading.Lock()


class TicketRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(default="", max_length=8000)
    sources: List[str] = []
    name: str = Field(default="", max_length=120)
    email: str = Field(default="", max_length=200)
    details: str = Field(default="", max_length=4000)
    priority: str = Field(default="medium", pattern="^(low|medium|high)$")


def _load() -> List[Dict[str, Any]]:
    if not TICKETS_FILE.exists():
        return []
    try:
        return json.loads(TICKETS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def store_ticket(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Built-in webhook receiver logic: persist the ticket and return its id."""
    with _lock:
        tickets = _load()
        ticket = {
            "ticket_id": f"IT-{1001 + len(tickets)}",
            "status": "open",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **payload,
        }
        tickets.append(ticket)
        TICKETS_FILE.write_text(json.dumps(tickets, indent=2, ensure_ascii=False), encoding="utf-8")
    return ticket


def list_tickets() -> List[Dict[str, Any]]:
    with _lock:
        return _load()


def send_to_webhook(req: TicketRequest) -> Dict[str, Any]:
    """POST the ticket to the configured webhook and return its JSON response."""
    payload = {
        "event": "ticket.create",
        "title": f"IT HelpDesk: {req.question[:100]}",
        "question": req.question,
        "assistant_answer": req.answer,
        "sources": req.sources,
        "requester": {"name": req.name, "email": req.email},
        "details": req.details,
        "priority": req.priority,
    }
    http_req = urllib.request.Request(
        WEBHOOK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Webhook-Secret": WEBHOOK_SECRET},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_req, timeout=15) as resp:
            body = resp.read().decode("utf-8") or "{}"
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Webhook rejected the ticket (HTTP {exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach the ticket webhook: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {}


def update_status(ticket_id: str, status: str) -> Optional[Dict[str, Any]]:
    with _lock:
        tickets = _load()
        for ticket in tickets:
            if ticket["ticket_id"] == ticket_id:
                ticket["status"] = status
                TICKETS_FILE.write_text(json.dumps(tickets, indent=2, ensure_ascii=False), encoding="utf-8")
                return ticket
    return None
