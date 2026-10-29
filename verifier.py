"""Claim Verifier: fact-checks the agent's own answer using fresh web searches.

Pipeline:
1. Extract the key factual claims from the answer.
2. Search the web for evidence for each claim.
3. Ask the model to judge each claim against the evidence.
4. Build a markdown report with verdicts and an overall reliability score.
"""

import json
import re

MAX_CLAIMS = 3
MAX_CLAIM_CHARS = 200
MAX_EVIDENCE_CHARS = 1000
MIN_ANSWER_CHARS = 80

VERDICT_ICONS = {
    "supported": "✅ Supported",
    "uncertain": "⚠️ Uncertain",
    "contradicted": "❌ Contradicted",
}

EXTRACT_PROMPT = """Extract up to {limit} key factual claims from the text below
that can be checked on the web (dates, numbers, names, events, scientific facts).
Skip opinions and vague statements. Each claim must be a short, self-contained sentence.

Return ONLY a JSON array of strings, nothing else.

Text:
{text}
"""

JUDGE_PROMPT = """You are a strict fact-checker.

Claim: {claim}

Web evidence:
{evidence}

Decide if the evidence supports the claim.
- "supported": the evidence clearly backs the claim.
- "contradicted": the evidence clearly conflicts with the claim.
- "uncertain": the evidence is missing, unrelated, or not conclusive.

Return ONLY a JSON object like:
{{"verdict": "supported", "confidence": 85, "reason": "max 20 words"}}
"""


def _text(response):
    """Get plain text from a LangChain response."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def _parse_json(raw, opener, closer):
    """Pull the first JSON array/object out of model text."""
    start = raw.find(opener)
    end = raw.rfind(closer)

    if start == -1 or end <= start:
        return None

    try:
        return json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def extract_claims(answer, llm):
    """Ask the model for the key checkable claims in the answer."""
    prompt = EXTRACT_PROMPT.format(limit=MAX_CLAIMS, text=answer[:2000])
    data = _parse_json(_text(llm.invoke(prompt)), "[", "]")

    if not isinstance(data, list):
        return []

    claims = []

    for item in data:
        if isinstance(item, str) and item.strip():
            claims.append(item.strip()[:MAX_CLAIM_CHARS])

    return claims[:MAX_CLAIMS]


def check_claim(claim, llm, search_tool):
    """Search for evidence and judge a single claim."""
    try:
        evidence = str(search_tool.invoke({"query": claim}))[:MAX_EVIDENCE_CHARS]
    except Exception as exc:
        evidence = f"Web search failed: {type(exc).__name__}"

    no_evidence = (
        not evidence.strip()
        or evidence.startswith("Web search failed")
        or evidence.startswith("No search results")
    )

    if no_evidence:
        return {
            "claim": claim,
            "verdict": "uncertain",
            "confidence": 0,
            "reason": "No web evidence could be retrieved.",
        }

    prompt = JUDGE_PROMPT.format(claim=claim, evidence=evidence)
    data = _parse_json(_text(llm.invoke(prompt)), "{", "}")

    if not isinstance(data, dict):
        return {
            "claim": claim,
            "verdict": "uncertain",
            "confidence": 0,
            "reason": "The verifier could not judge this claim.",
        }

    verdict = str(data.get("verdict", "uncertain")).lower().strip()

    if verdict not in VERDICT_ICONS:
        verdict = "uncertain"

    try:
        confidence = int(float(data.get("confidence", 0)))
    except (TypeError, ValueError):
        confidence = 0

    confidence = max(0, min(100, confidence))

    return {
        "claim": claim,
        "verdict": verdict,
        "confidence": confidence,
        "reason": str(data.get("reason", "")).strip()[:160],
    }


def _cell(text):
    """Make text safe for a markdown table cell."""
    return str(text).replace("|", "/").replace("\n", " ").strip()


def build_report(results):
    """Turn claim results into a markdown report."""
    total = len(results)
    supported = sum(1 for r in results if r["verdict"] == "supported")
    contradicted = sum(1 for r in results if r["verdict"] == "contradicted")
    score = round(100 * supported / total) if total else 0

    if contradicted:
        status = "🔴 Some claims were contradicted. Double-check this answer."
    elif supported == total:
        status = "🟢 All checked claims are supported by web evidence."
    else:
        status = "🟡 Partly verified. Some claims need manual checking."

    lines = [
        "---",
        "### 🔍 Claim Verification",
        f"**Reliability: {supported}/{total} claims supported ({score}%)**  ",
        status,
        "",
        "| # | Claim | Verdict | Confidence | Note |",
        "|---|-------|---------|------------|------|",
    ]

    for index, item in enumerate(results, start=1):
        lines.append(
            f"| {index} | {_cell(item['claim'])} "
            f"| {VERDICT_ICONS[item['verdict']]} "
            f"| {item['confidence']}% "
            f"| {_cell(item['reason'])} |"
        )

    return "\n".join(lines)


def verify_answer(answer, llm, search_tool):
    """Fact-check an answer. Returns a markdown report, or '' if nothing to check."""
    if not answer or len(answer.strip()) < MIN_ANSWER_CHARS:
        return ""

    try:
        claims = extract_claims(answer, llm)

        if not claims:
            return ""

        results = [check_claim(claim, llm, search_tool) for claim in claims]

        return build_report(results)

    except Exception as exc:
        print("VERIFIER ERROR:", repr(exc))
        return (
            "---\n"
            "### 🔍 Claim Verification\n"
            "Verification could not be completed "
            "(possibly a Groq rate limit). Try again in a moment."
        )