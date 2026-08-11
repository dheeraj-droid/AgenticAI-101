"""The FastAPI app behind ``onboarding serve``.

One page, three endpoints, localhost only. Every route is a thin wrapper over
``web.service`` — the same discipline the framework adapters follow, for the
same reason: the interesting behaviour should be testable without a server.

Binding: ``127.0.0.1`` by default, deliberately. This app runs real onboarding
with real mail, and nothing about it is hardened for exposure to a network.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from onboarding.adapters.base import FRAMEWORKS, get_adapter
from onboarding.core.errors import OnboardingError
from onboarding.core.llm import llm_spec
from onboarding.core.mailer import allowlist, smtp_configured, support_address
from onboarding.core.registry import registry_path
from onboarding.web import service
from onboarding.web.models import ChatRequest, ChatResponse, OnboardRequest, OnboardResponse


def create_app(*, allow_send: bool = False) -> FastAPI:
    app = FastAPI(title="Customer Onboarding Assistant", docs_url=None, redoc_url=None)
    demo = service.Demo(allow_send=allow_send)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return service.page()

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        """What the page needs to tell the user what will really happen."""
        spec = llm_spec()
        approved = allowlist()
        return {
            "frameworks": [
                {
                    "id": name,
                    "label": _LABELS[name][0],
                    "full_name": _LABELS[name][1],
                    "note": get_adapter(name).capabilities.notes,
                }
                for name in FRAMEWORKS
            ],
            "llm": {"configured": spec.configured, "model": spec.model or ""},
            "mail": {
                "sending": allow_send and smtp_configured(),
                "smtp_configured": smtp_configured(),
                "allow_send": allow_send,
                "support_address": support_address(),
                "approved_recipients": sorted(approved),
            },
            "registry_path": str(registry_path()),
        }

    @app.post("/api/onboard", response_model=OnboardResponse)
    async def onboard(request: OnboardRequest) -> OnboardResponse:
        if request.framework not in FRAMEWORKS:
            raise HTTPException(400, f"unknown framework {request.framework!r}")
        try:
            return await demo.onboard(request)
        except OnboardingError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            return await demo.ask(request.session_id, request.question)
        except KeyError as exc:
            raise HTTPException(404, "that chat session has expired — onboard again") from exc
        except OnboardingError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


# (tab label, full name). The tab strip has four equal columns, so the label has
# to fit one of them; the full name goes in the tooltip.
_LABELS = {
    "maf": ("Agent Framework", "Microsoft Agent Framework"),
    "langchain": ("LangChain", "LangChain"),
    "langgraph": ("LangGraph", "LangGraph"),
    "crew": ("CrewAI", "CrewAI"),
}


def serve(host: str = "127.0.0.1", port: int = 8000, *, allow_send: bool = False) -> None:
    """Run the page. Blocks until interrupted."""
    import uvicorn

    os.environ.setdefault("ONBOARDING_WEB", "1")
    uvicorn.run(create_app(allow_send=allow_send), host=host, port=port, log_level="warning")
