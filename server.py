"""IT HelpDesk Assistant: UI + LangServe API on a single URL.

    UI                http://localhost:8000/
    LangServe API     http://localhost:8000/helpdesk/invoke | /stream | /playground
"""
from pathlib import Path

import secrets

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langserve import add_routes
from pydantic import BaseModel, Field

from rag_chain import build_chain
from tickets import WEBHOOK_SECRET, TicketRequest, list_tickets, send_to_webhook, store_ticket, update_status

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="IT HelpDesk Assistant")

add_routes(app, build_chain(), path="/helpdesk")


@app.post("/api/tickets")
def create_ticket(req: TicketRequest):
    """Called by the UI after the user agrees to open a ticket; forwards it to the webhook."""
    try:
        result = send_to_webhook(req)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"ticket_id": result.get("ticket_id", "submitted"), "status": result.get("status", "open")}


@app.get("/api/tickets")
def get_tickets():
    """Newest first. Used by the /tickets dashboard."""
    return {"tickets": list(reversed(list_tickets()))}


class StatusUpdate(BaseModel):
    status: str = Field(pattern="^(open|in_progress|resolved)$")


@app.patch("/api/tickets/{ticket_id}")
def set_ticket_status(ticket_id: str, body: StatusUpdate):
    ticket = update_status(ticket_id, body.status)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return ticket


@app.get("/tickets", include_in_schema=False)
def tickets_page():
    return FileResponse(STATIC_DIR / "tickets.html")


@app.post("/webhook/tickets", include_in_schema=False)
async def webhook_receiver(request: Request, x_webhook_secret: str = Header(default="")):
    """Built-in webhook target: stores tickets in tickets.json. Replace via TICKET_WEBHOOK_URL."""
    if not secrets.compare_digest(x_webhook_secret, WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid webhook secret.")
    ticket = store_ticket(await request.json())
    return {"ticket_id": ticket["ticket_id"], "status": ticket["status"]}
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
