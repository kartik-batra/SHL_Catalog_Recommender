"""
Local evaluation harness.

Simulates the SHL automated evaluator by:
1. Loading a trace file (JSON with persona + expected_assessments).
2. Running a multi-turn conversation against POST /chat.
3. Computing Recall@10 and reporting behavior-probe results.

Usage:
    python scripts/evaluate.py --traces data/traces/ --url http://localhost:8000
    python scripts/evaluate.py --traces data/traces/trace_001.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────── helpers ─────────────────────────────────────────


def recall_at_k(recommended: list[str], relevant: list[str], k: int = 10) -> float:
    """Fraction of relevant assessments present in the top-k recommendations."""
    if not relevant:
        return 1.0  # vacuously true
    top_k_names = {n.lower() for n in recommended[:k]}
    hits = sum(1 for r in relevant if r.lower() in top_k_names)
    return hits / len(relevant)


def run_trace(
    trace: dict,
    base_url: str,
    timeout: int = 30,
) -> dict:
    """
    Run a single conversation trace.

    Expected trace format:
    {
      "id": "trace_001",
      "persona": "...",          // description for the simulated user (unused here)
      "opening": "...",          // first user message
      "facts": {...},            // answers to give if asked (unused — handled by user sim)
      "expected_assessments": ["Name A", "Name B"]  // ground truth
    }
    """
    result = {
        "id": trace.get("id", "unknown"),
        "turns": 0,
        "schema_violations": 0,
        "hallucinated_urls": 0,
        "recommended_on_turn_1_for_vague": False,
        "final_recommendations": [],
        "recall_at_10": 0.0,
        "passed_turn_cap": True,
    }

    messages: list[dict] = []
    opening = trace.get("opening", trace.get("persona", "I need an assessment."))
    messages.append({"role": "user", "content": opening})

    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for turn in range(8):
            payload = {"messages": messages}

            try:
                resp = client.post("/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.error("Request failed on turn %d: %s", turn + 1, exc)
                result["schema_violations"] += 1
                break

            # ── Schema check ──────────────────────────────────────────────────
            for required_field in ("reply", "recommendations", "end_of_conversation"):
                if required_field not in data:
                    logger.warning(
                        "SCHEMA VIOLATION: missing field '%s' on turn %d",
                        required_field,
                        turn + 1,
                    )
                    result["schema_violations"] += 1

            reply = data.get("reply", "")
            recs = data.get("recommendations", [])
            eoc = data.get("end_of_conversation", False)
            result["turns"] = turn + 1

            # ── Behavior probe: no recommendation on turn 1 for vague query ──
            if turn == 0 and recs:
                vague_keywords = {"assessment", "test", "something", "help", "need"}
                first_words = set(opening.lower().split())
                if first_words & vague_keywords and len(opening.split()) < 8:
                    logger.warning(
                        "BEHAVIOR PROBE FAIL: recommended on turn 1 for vague query."
                    )
                    result["recommended_on_turn_1_for_vague"] = True

            # ── Turn cap check ────────────────────────────────────────────────
            if turn + 1 > 8:
                result["passed_turn_cap"] = False

            # ── Record final recommendations ──────────────────────────────────
            if recs:
                result["final_recommendations"] = [r.get("name", "") for r in recs]

            logger.info(
                "Turn %d | intent inferred | recs=%d | eoc=%s | reply=%.80s",
                turn + 1,
                len(recs),
                eoc,
                reply,
            )

            if eoc or recs:
                break  # conversation complete

            # Simulate user: echo the agent's clarifying question with a canned answer
            # (In production the LLM user sim answers from its facts sheet.)
            facts = trace.get("facts", {})
            simulated_answer = (
                f"I have no specific preference on that. "
                f"The role is: {facts.get('role', 'software developer')}. "
                f"Level: {facts.get('level', 'mid-level')}."
            )
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": simulated_answer})

            time.sleep(0.5)  # avoid rate-limiting

    # ── Recall@10 ─────────────────────────────────────────────────────────────
    expected = trace.get("expected_assessments", [])
    result["recall_at_10"] = recall_at_k(
        result["final_recommendations"], expected, k=10
    )

    return result


# ─────────────────────────── main ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SHL recommender.")
    parser.add_argument(
        "--traces",
        default="data/traces",
        help="Path to a single trace JSON file or a directory of trace files.",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL of the running service."
    )
    args = parser.parse_args()

    trace_path = Path(args.traces)
    if trace_path.is_file():
        trace_files = [trace_path]
    elif trace_path.is_dir():
        trace_files = sorted(trace_path.glob("*.json"))
    else:
        logger.error("Trace path not found: %s", trace_path)
        sys.exit(1)

    if not trace_files:
        logger.error("No trace files found in %s", trace_path)
        sys.exit(1)

    # ── Check service health ───────────────────────────────────────────────────
    try:
        with httpx.Client(base_url=args.url, timeout=120) as client:
            health = client.get("/health")
            health.raise_for_status()
            logger.info("Service healthy: %s", health.json())
    except Exception as exc:
        logger.error("Cannot reach service at %s: %s", args.url, exc)
        sys.exit(1)

    # ── Run traces ────────────────────────────────────────────────────────────
    results = []
    for tf in trace_files:
        with open(tf) as f:
            trace = json.load(f)
        logger.info("=" * 60)
        logger.info("Running trace: %s", tf.name)
        result = run_trace(trace, base_url=args.url)
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    mean_recall = sum(r["recall_at_10"] for r in results) / len(results)
    schema_fails = sum(r["schema_violations"] > 0 for r in results)
    turn_cap_fails = sum(not r["passed_turn_cap"] for r in results)
    vague_turn1_fails = sum(r["recommended_on_turn_1_for_vague"] for r in results)

    for r in results:
        logger.info(
            "%-20s  Recall@10=%.2f  turns=%d  schema_ok=%s",
            r["id"],
            r["recall_at_10"],
            r["turns"],
            r["schema_violations"] == 0,
        )

    logger.info("-" * 60)
    logger.info("Mean Recall@10      : %.3f", mean_recall)
    logger.info("Schema violations   : %d / %d traces", schema_fails, len(results))
    logger.info("Turn cap violations : %d / %d traces", turn_cap_fails, len(results))
    logger.info("Vague-turn-1 fails  : %d / %d traces", vague_turn1_fails, len(results))


if __name__ == "__main__":
    main()
