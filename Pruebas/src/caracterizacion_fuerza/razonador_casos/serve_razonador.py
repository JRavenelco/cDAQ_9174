#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve_razonador.py — FastAPI microservice for the CBR Reasoner.

Endpoints:
  GET  /health              → Service status + case count
  GET  /cases               → List all cases in the library
  GET  /cases/{case_id}     → Get a specific case by ID
  POST /retrieve            → Retrieve top-K similar cases (no LLM)
  POST /reason              → Full CBR pipeline: retrieve + LLM reasoning
  POST /reason/features     → Reason from raw features dict (no case_id needed)

Run:
  python serve_razonador.py                          # default port 8100
  python serve_razonador.py --port 8200 --model gemma4
  python serve_razonador.py --reload                 # dev mode with auto-reload

n8n Integration:
  Use "HTTP Request" node → POST http://<host>:8100/reason
  Body: {"case_id": "corte_20260311_131921_600_40", "top_k": 3}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Import core logic from razonador_local
from razonador_local import (
    CASES_JSONL,
    DEFAULT_MODEL,
    FEATURE_WEIGHTS,
    OLLAMA_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OUTPUT_DIR,
    PROMPT_FILE,
    RETRIEVAL_FEATURES,
    as_float,
    build_reasoning_prompt,
    call_llm,
    resolve_provider,
    load_cases,
    load_system_prompt,
    retrieve_similar,
    save_result,
)

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Razonador CBR - Diagnóstico de Histéresis",
    description=(
        "Case-Based Reasoning service for hysteresis and tool wear diagnosis. "
        "Retrieves similar historical cutting cases and reasons about them "
        "using a local LLM (Ollama)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────────

_library: list[dict[str, Any]] = []
_system_prompt: str = ""
_config: dict[str, Any] = {}


# ── Pydantic models ───────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    case_id: Optional[str] = Field(None, description="ID of existing case in library")
    features: Optional[dict[str, float]] = Field(None, description="Raw feature dict (alternative to case_id)")
    top_k: int = Field(5, ge=1, le=19, description="Number of similar cases to retrieve")

class ReasonRequest(BaseModel):
    case_id: Optional[str] = Field(None, description="ID of existing case in library")
    features: Optional[dict[str, float]] = Field(None, description="Raw feature dict (alternative to case_id)")
    top_k: int = Field(5, ge=1, le=19, description="Number of similar cases to retrieve")
    model: Optional[str] = Field(None, description="Ollama model override")
    temperature: float = Field(0.3, ge=0.0, le=1.0, description="LLM temperature")

class MatchOut(BaseModel):
    case_id: str
    score: float
    distance: float
    loop_area_norm: float
    rpm_estimada: Any
    source_kind: str

class RetrieveResponse(BaseModel):
    query_case_id: str
    query_features: dict[str, float]
    matches: list[MatchOut]
    retrieval_time_ms: float

class ReasonResponse(BaseModel):
    case_id: str
    query_features: dict[str, float]
    matches: list[MatchOut]
    reasoning: str
    model: str
    retrieval_time_ms: float
    reasoning_time_s: float
    total_time_s: float

class CaseSummary(BaseModel):
    case_id: str
    source_kind: str
    loop_area_norm: float
    force_rms: float
    rpm_estimada: Any
    duration_s: float

class HealthResponse(BaseModel):
    status: str
    cases_loaded: int
    model: str
    ollama_url: str
    cloud_available: bool
    cloud_model: str


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve_query(case_id: Optional[str], features: Optional[dict]) -> dict:
    """Resolve a query case from case_id or raw features."""
    if case_id:
        found = [c for c in _library if c.get("case_id") == case_id]
        if not found:
            available = [c["case_id"] for c in _library]
            raise HTTPException(
                status_code=404,
                detail=f"case_id '{case_id}' not found. Available: {available}",
            )
        return found[0]
    elif features:
        # Build a minimal case dict from features
        case = {"case_id": features.get("case_id", "query_adhoc"), "source_kind": "api_query"}
        case.update(features)
        return case
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'case_id' or 'features'.",
        )


def _do_retrieve(query: dict, top_k: int) -> tuple[list, float]:
    """Run retrieval and return matches + timing."""
    t0 = time.time()
    matches = retrieve_similar(
        query=query,
        library=_library,
        features=RETRIEVAL_FEATURES,
        weights=FEATURE_WEIGHTS,
        top_k=top_k,
    )
    elapsed = (time.time() - t0) * 1000
    return matches, elapsed


def _matches_to_dicts(matches) -> list[dict]:
    return [
        {
            "case_id": m.case.get("case_id", ""),
            "score": round(m.score, 6),
            "distance": round(m.distance, 6),
            "loop_area_norm": as_float(m.case, "loop_area_norm"),
            "rpm_estimada": m.case.get("rpm_estimada", ""),
            "source_kind": m.case.get("source_kind", ""),
        }
        for m in matches
    ]


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        cases_loaded=len(_library),
        model=_config.get("model", DEFAULT_MODEL),
        ollama_url=OLLAMA_URL,
        cloud_available=bool(OPENROUTER_API_KEY),
        cloud_model=OPENROUTER_MODEL,
    )


@app.get("/cases", response_model=list[CaseSummary])
def list_cases():
    return [
        CaseSummary(
            case_id=c.get("case_id", ""),
            source_kind=c.get("source_kind", ""),
            loop_area_norm=as_float(c, "loop_area_norm"),
            force_rms=as_float(c, "force_rms"),
            rpm_estimada=c.get("rpm_estimada", ""),
            duration_s=as_float(c, "duration_s"),
        )
        for c in _library
    ]


@app.get("/cases/{case_id}")
def get_case(case_id: str):
    found = [c for c in _library if c.get("case_id") == case_id]
    if not found:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")
    return found[0]


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    query = _resolve_query(req.case_id, req.features)
    matches, elapsed = _do_retrieve(query, req.top_k)

    return RetrieveResponse(
        query_case_id=query.get("case_id", ""),
        query_features={f: as_float(query, f) for f in RETRIEVAL_FEATURES},
        matches=_matches_to_dicts(matches),
        retrieval_time_ms=round(elapsed, 1),
    )


@app.post("/reason", response_model=ReasonResponse)
def reason(req: ReasonRequest):
    query = _resolve_query(req.case_id, req.features)
    model = req.model or _config.get("model", DEFAULT_MODEL)

    t_total = time.time()

    # Retrieve
    matches, retrieval_ms = _do_retrieve(query, req.top_k)

    # Build prompt and call LLM (local Ollama o cloud OpenRouter segun el modelo)
    user_prompt = build_reasoning_prompt(query, matches)
    t_llm = time.time()
    reasoning = call_llm(user_prompt, _system_prompt, model, req.temperature)
    reasoning_s = time.time() - t_llm

    total_s = time.time() - t_total

    # Save result
    result_dict = {
        "case_id": query.get("case_id", ""),
        "query_features": {f: as_float(query, f) for f in RETRIEVAL_FEATURES},
        "top_matches": _matches_to_dicts(matches),
        "reasoning": reasoning,
        "model": model,
        "retrieval_time_ms": round(retrieval_ms, 1),
        "reasoning_time_s": round(reasoning_s, 2),
        "total_time_s": round(total_s, 2),
    }
    try:
        save_result(result_dict, OUTPUT_DIR)
    except Exception:
        pass  # Non-critical

    return ReasonResponse(
        case_id=query.get("case_id", ""),
        query_features={f: as_float(query, f) for f in RETRIEVAL_FEATURES},
        matches=_matches_to_dicts(matches),
        reasoning=reasoning,
        model=model,
        retrieval_time_ms=round(retrieval_ms, 1),
        reasoning_time_s=round(reasoning_s, 2),
        total_time_s=round(total_s, 2),
    )


# ── Startup ────────────────────────────────────────────────────────────────

def init_app(cases_file: Path = CASES_JSONL, model: str = DEFAULT_MODEL):
    """Load case library and system prompt."""
    global _library, _system_prompt, _config
    _library = load_cases(cases_file)
    _system_prompt = load_system_prompt(PROMPT_FILE)
    _config["model"] = model
    print(f"[CBR] {len(_library)} cases loaded from {cases_file.name}")
    print(f"[CBR] Model: {model} | Ollama: {OLLAMA_URL}")
    print(f"[CBR] System prompt: {len(_system_prompt)} chars")


@app.on_event("startup")
async def on_startup():
    if not _library:
        init_app()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CBR Reasoner FastAPI Service")
    parser.add_argument("--port", type=int, default=8100, help="Port (default 8100)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default 0.0.0.0)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    parser.add_argument("--cases-file", type=Path, default=CASES_JSONL, help="JSONL cases file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    args = parser.parse_args()

    init_app(cases_file=args.cases_file, model=args.model)

    print(f"\n{'='*60}")
    print(f"  CBR Reasoner → http://{args.host}:{args.port}")
    print(f"  Docs         → http://localhost:{args.port}/docs")
    print(f"{'='*60}\n")

    uvicorn.run(
        "serve_razonador:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
