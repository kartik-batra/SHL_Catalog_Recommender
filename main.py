"""
FastAPI service for the SHL Assessment Recommender.

Endpoints:
    GET  /health  →  {"status": "ok"}
    POST /chat    →  ChatResponse

The FAISS index and FastEmbed model are loaded once at startup so cold-start
latency (Render free tier, ~2 min) amortises across all requests.

Agent.run() is CPU/IO-bound but fast enough to run in the default thread pool;
we wrap it with asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.agent import Agent
from src.config import settings
from src.models import ChatRequest, ChatResponse
from src.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────── lifespan ────────────────────────────────────────

_retriever: Retriever | None = None
_agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global _retriever, _agent

    logger.info("Starting up — loading index and embedding model…")
    start = time.perf_counter()
    try:
        _retriever = Retriever(
            index_path=settings.index_path,
            meta_path=settings.meta_path,
            model_name=settings.embed_model,
        )
        _agent = Agent(retriever=_retriever)
    except FileNotFoundError as exc:
        logger.error("Startup failed: %s", exc)
        logger.error(
            "Run 'python scripts/build_catalog.py' to generate the index first."
        )
        raise

    elapsed = time.perf_counter() - start
    logger.info(
        "Ready in %.1fs — %d assessments indexed.", elapsed, len(_retriever.catalog)
    )
    yield
    logger.info("Shutting down.")


# ─────────────────────────── app ─────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent that recommends SHL Individual Test Solutions "
        "from the official product catalog."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────── middleware: timing ───────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[type-arg]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s → %d  (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# ─────────────────────────── exception handler ───────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ─────────────────────────── endpoints ───────────────────────────────────────

@app.get("/health", summary="Readiness probe")
async def health() -> dict:
    """Returns 200 once the index is loaded and the service is ready."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Service initialising")
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a conversation turn and receive the next agent reply",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless endpoint.  The caller must send the FULL conversation history on
    every request.  The service holds no per-conversation state.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Service initialising")

    messages = [msg.model_dump() for msg in request.messages]

    # Run the (synchronous) agent in a thread pool to avoid blocking the event loop
    try:
        response: ChatResponse = await asyncio.to_thread(_agent.run, messages)
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        return ChatResponse(
            reply=(
                "I encountered an unexpected error. "
                "Please try again or rephrase your request."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    return response
