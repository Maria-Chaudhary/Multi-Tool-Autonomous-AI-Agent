"""Multi-tool Autonomous AI Research Agent with Claim Verifier.

Control-panel UI: sidebar, live agent-brain dashboard, action cards,
stats row, activity log and the chat workspace.
"""
import html
import inspect
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import NamedTuple

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq

from tools import tools, web_search, read_uploaded_file
from verifier import verify_answer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

try:
    CUSTOM_CSS = (BASE_DIR / "style.css").read_text(encoding="utf-8")
except OSError:
    print("WARNING: style.css not found. The app will use default styling.")
    CUSTOM_CSS = ""


# ---------------- CONFIGURATION ----------------

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

MAX_QUESTION_CHARS = 1500
MAX_DOCUMENT_CHARS = 2500
MAX_TOOL_RESULT_CHARS = 1200
MAX_SEARCH_RESULT_CHARS = 2600
MAX_FINAL_RESULTS_CHARS = 4000
MAX_TOOL_ROUNDS = 2
MAX_TOOLS_PER_ROUND = 2
MAX_REPLY_TOKENS = 900
MAX_HISTORY_MESSAGES = 4
MAX_HISTORY_CHARS = 700

# "low" makes gpt-oss answer much faster and leaves more tokens for the
# visible reply. Use low / medium / high, or "off" to disable.
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low").strip().lower()


class Budget(NamedTuple):
    """How much work one question may use."""
    reply_tokens: int
    tool_rounds: int
    tools_per_round: int
    final_results_chars: int


# Normal questions: quick, small answers.
SIMPLE_BUDGET = Budget(
    reply_tokens=MAX_REPLY_TOKENS,
    tool_rounds=MAX_TOOL_ROUNDS,
    tools_per_round=MAX_TOOLS_PER_ROUND,
    final_results_chars=MAX_FINAL_RESULTS_CHARS,
)

# Research questions (several items, comparisons, recommendations):
# several searches at once, then one long final answer.
DEEP_BUDGET = Budget(
    reply_tokens=2600,
    tool_rounds=1,
    tools_per_round=5,
    final_results_chars=6500,
)

# Estimated manual research time saved per answered question (seconds).
SECONDS_SAVED_PER_ANSWER = 600
MAX_LOG_ENTRIES = 8

SYSTEM_PROMPT = """
You are a helpful autonomous AI research assistant.

You can use web_search to find information and calculator
to perform basic calculations.

Instructions:
- Answer clearly and accurately.
- Use tools only when necessary.
- Summarize search results instead of repeating them.
- Use uploaded document text when answering document questions.
- Keep answers concise and relevant.
- Do not invent sources, URLs, references, or facts.
- Answer EVERY part of the request. If the user asks for N items, give
  all N, then the comparison and the recommendation they asked for.
- If the user asks you to research, search, or provide sources, you MUST
  call web_search before answering.
- For a request with several topics, call web_search once per topic in the
  same step (up to 5 calls), each with a short specific query.

TOOL USAGE RULES:

1. When using web_search, provide exactly one argument:
   {"query": "your search query"}

2. Never provide cursor, id, page, or other arguments
   to web_search.

3. When using calculator, provide exactly:
   {"expression": "your mathematical expression"}

4. Do not invent tool names or tool arguments.

5. Use the results returned by the tools.
   Do not attempt to paginate or request additional
   search results using cursor or id.

6. When sufficient information has been collected,
   provide the final answer without requesting more tools.


SOURCE CITATION RULES:
- When web search results are available, cite important factual
  claims using the exact reference IDs provided in the results.
- Use citation format like [1], [2], or [3].
- Place citations immediately after the claims they support.
- Only cite sources that actually support the claim.
- Never invent reference IDs.
- Do not create fake URLs.
- If no web sources are available, do not add web citations.
"""


def build_system_prompt(extra=""):
    """System prompt with today's date, so searches use the right year."""
    return (
        SYSTEM_PROMPT
        + f"\nToday's date is {date.today().isoformat()}.\n"
        + extra
    )


# ---------------- MODEL SETUP ----------------

_base_model = None
_tool_model = None


def get_base_model():
    """Model without tools (final answers and claim verification)."""
    global _base_model

    if _base_model is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Set it in your environment "
                "or .env file."
            )

        model_kwargs = dict(
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=MAX_REPLY_TOKENS,
            api_key=api_key,
            timeout=90,
            max_retries=2,
        )

        if "gpt-oss" in MODEL_NAME.lower() and REASONING_EFFORT in (
            "low", "medium", "high"
        ):
            model_kwargs["reasoning_effort"] = REASONING_EFFORT

        _base_model = ChatGroq(**model_kwargs)

    return _base_model


def get_tool_model():
    """Model with tools bound (used inside the agent loop)."""
    global _tool_model

    if _tool_model is None:
        _tool_model = get_base_model().bind_tools(tools)

    return _tool_model


# ---------------- HELPER FUNCTIONS ----------------

SOURCES_MARKER = "\n\n---\n### 🔗 Sources & References"
VERIFY_MARKER = "\n\n---\n### 🔍 Claim Verification"


def content_to_text(content):
    """Convert any Gradio/LangChain content format into plain text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        if "text" in content:
            return str(content["text"])
        return ""

    if isinstance(content, (list, tuple)):
        parts = []

        for item in content:
            text = content_to_text(item)
            if text:
                parts.append(text)

        return "\n".join(parts)

    return str(content)


def strip_verification(text):
    """Remove the verification report and source list from old answers,
    so they are not sent back to the model."""
    cut = len(text)

    for marker in (VERIFY_MARKER, SOURCES_MARKER):
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)

    return text[:cut]


def get_recent_messages(history):
    """Turn recent chat history into LangChain messages."""
    messages = []

    for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = strip_verification(content_to_text(item.get("content", "")))
        content = content[:MAX_HISTORY_CHARS]

        if not content:
            continue

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def normalize_history(history):
    """
    Return history as a list of {"role", "content"} dictionaries.
    Also accepts old tuple-style history and converts it.
    """
    clean = []

    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")

            if role in ("user", "assistant"):
                clean.append({
                    "role": role,
                    "content": content_to_text(item.get("content", "")),
                })

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            if item[0] is not None:
                clean.append({"role": "user", "content": str(item[0])})
            if item[1] is not None:
                clean.append({"role": "assistant", "content": str(item[1])})

    return clean


def is_tool_failure(exc):
    """True when Groq rejected a malformed or unexpected tool call."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "tool_use_failed",
            "tool choice is none",
            "failed_generation",
            "tool call validation",
            "tool_call",
        )
    )


def friendly_error(exc):
    """Turn raw API errors into short, readable messages."""
    text = str(exc)
    low = text.lower()

    if "GROQ_API_KEY" in text:
        return text

    if (
        "413" in text
        or "429" in text
        or "tokens per minute" in low
        or "rate_limit_exceeded" in low
        or "rate limit" in low
    ):
        return (
            "Groq's rate or token limit was reached. Wait a moment, or "
            "shorten your question or document, then try again."
        )

    if "401" in text or "invalid api key" in low:
        return "Groq rejected the API key. Check GROQ_API_KEY and try again."

    if is_tool_failure(exc):
        return (
            "The model produced an invalid tool call. "
            "Please try again or rephrase your question."
        )

    if "timeout" in low or "timed out" in low or "connection" in low:
        return (
            "Could not reach Groq. Check your internet connection "
            "and try again."
        )

    return f"An error occurred: {text}"



def was_truncated(response):
    """True when the model stopped because it ran out of tokens."""
    meta = getattr(response, "response_metadata", None) or {}
    return meta.get("finish_reason") == "length"


def rate_limit_wait(exc):
    """Seconds to wait before retrying a Groq rate limit, else None."""
    text = str(exc)
    low = text.lower()

    if "413" in text or "429" not in text and "rate_limit" not in low:
        return None

    match = re.search(
        r"try again in\s*(?:(\d+)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?\s*(?:(\d+(?:\.\d+)?)ms)?",
        low,
    )
    wait = 5.0

    if match and any(match.groups()):
        minutes, seconds, millis = match.groups()
        wait = (
            float(minutes or 0) * 60
            + float(seconds or 0)
            + float(millis or 0) / 1000
            + 0.5
        )

    # Do not make the user wait minutes (for example daily limits).
    return wait if wait <= 30 else None


def invoke_with_retry(model, messages, **kwargs):
    """Call the model; wait and retry when Groq's per-minute limit is hit."""
    for attempt in range(3):
        try:
            return model.invoke(messages, **kwargs)
        except Exception as exc:
            wait = rate_limit_wait(exc)

            if wait is None or attempt == 2:
                raise

            time.sleep(wait)


RESEARCH_WORDS = re.compile(
    r"\b(search|sources?|references?|research|compare|comparison|"
    r"strategies|recommend|analy[sz]e|analysis|pros and cons|report)\b",
    re.IGNORECASE,
)
COUNTED_ITEMS = re.compile(
    r"\b(\d+|three|four|five|six|seven|eight|nine|ten)\b[^.?!]{0,40}"
    r"\b(ways|strategies|tips|options|methods|ideas|tools|steps)\b",
    re.IGNORECASE,
)


def is_complex_request(question):
    """Detect research-style questions that need a long, multi-search reply."""
    text = question or ""
    hits = len(set(m.lower() for m in RESEARCH_WORDS.findall(text)))

    return bool(
        len(text) > 350
        or hits >= 3
        or (hits >= 1 and COUNTED_ITEMS.search(text))
    )


# ---------------- TOOL EXECUTION ----------------

def sanitize_tool_args(tool_name, args):
    """Keep only the arguments each tool really accepts.

    Models sometimes invent extra arguments (cursor, id, page ...).
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {"query": args, "expression": args}

    if not isinstance(args, dict):
        args = {}

    def first_text(*keys):
        for key in keys:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in args.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    if tool_name == "web_search":
        return {"query": first_text("query", "q", "search")[:300]}

    if tool_name == "calculator":
        return {"expression": first_text("expression", "expr")[:200]}

    return args


def run_tool_call(tool_call):
    """Execute a single tool call and return its result."""

    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {}) or {}

    selected_tool = next(
        (item for item in tools if item.name == tool_name),
        None,
    )

    if selected_tool is None:
        return "Unknown tool requested."

    try:
        result = str(selected_tool.invoke(tool_args))

        # Structured web results are shortened later, result by result.
        if tool_name == "web_search":
            return result

        return result[:MAX_TOOL_RESULT_CHARS]

    except Exception as exc:
        return f"Tool failed: {type(exc).__name__}: {exc}"


def format_search_for_model(data, sources):
    """Turn structured search results into compact numbered text.

    Registers each shown result in `sources` with a stable reference ID.
    """
    query = data.get("query", "")
    results = data.get("results") or []

    if not results:
        reason = data.get("error") or "No search results found."
        return (
            f"Web search returned no results ({reason}). "
            "Answer without search results and say so."
        )

    header = f'Web search results for "{query}":'
    lines = [header]
    used = len(header)

    for item in results:
        url = str(item.get("url", "")).strip()

        if not url.startswith(("https://", "http://")):
            continue

        existing = next((s for s in sources if s["url"] == url), None)
        source_id = existing["id"] if existing else len(sources) + 1
        title = str(item.get("title") or "Untitled source").strip()
        snippet = str(item.get("snippet") or "").strip()

        block = f"[{source_id}] {title}\nURL: {url}\nSnippet: {snippet}"

        if used + len(block) > MAX_SEARCH_RESULT_CHARS:
            break

        if not existing:
            sources.append({"id": source_id, "title": title, "url": url})

        lines.append(block)
        used += len(block)

    if len(lines) == 1:
        return "Web search returned no usable results."

    return "\n\n".join(lines)


def prepare_tool_result(tool_name, result, sources):
    """Give the model clean text, and collect sources for the final list."""
    if tool_name != "web_search":
        return result

    try:
        data = json.loads(result)

        if isinstance(data, dict) and data.get("type") == "web_search_results":
            return format_search_for_model(data, sources)

    except (ValueError, TypeError, AttributeError):
        pass

    return str(result)[:MAX_TOOL_RESULT_CHARS]


# ---------------- AGENT ----------------

def complete_with_continuation(base_messages, max_tokens):
    """Call the model; if the reply is cut off, ask it to continue."""
    response = invoke_with_retry(
        get_base_model(), base_messages, max_tokens=max_tokens
    )
    text = content_to_text(response.content).strip()

    for _ in range(2):
        if not text or not was_truncated(response):
            break

        response = invoke_with_retry(
            get_base_model(),
            [
                *base_messages,
                AIMessage(content=text),
                HumanMessage(
                    content="Continue exactly where you stopped. Do not "
                            "repeat earlier text. Finish every remaining part."
                ),
            ],
            max_tokens=max_tokens,
        )
        more = content_to_text(response.content).strip()

        if not more:
            break

        text += (" " if text[-1].isalnum() else "\n") + more

    return text


def generate_final_answer(question, history, gathered, budget=SIMPLE_BUDGET):
    """Final answer WITHOUT tools.

    The earlier tool calls are NOT replayed here. If they were, the model
    would copy them and try to call a tool, which Groq rejects when no
    tools are bound. Results are passed as plain text instead.
    """
    if gathered:
        results_text = "\n\n".join(text for _, text in gathered)
        results_text = results_text[:budget.final_results_chars]

        instruction = (
            "Research results gathered so far:\n"
            f"{results_text}\n\n"
            "Answer every part of the request completely (all requested "
            "items, the comparison and the recommendation). Use a short "
            "Markdown table for comparisons. Cite web sources with their "
            "[n] reference IDs. If the results are not enough, say so. "
            "Do not call any tools."
        )
    else:
        results_text = ""
        instruction = (
            "No tool results are available. Answer every part of the "
            "request from your general knowledge, say clearly if you are "
            "unsure, and add no citations. Do not call any tools."
        )

    system = SystemMessage(
        content=build_system_prompt(
            "\nNo tools are available for this reply. Never call a tool. "
            "Reply in plain text."
        )
    )
    final_question = HumanMessage(content=f"{question}\n\n{instruction}")

    # Second attempt drops the chat history and allows a longer reply.
    attempts = (
        (get_recent_messages(history), budget.reply_tokens),
        ([], int(budget.reply_tokens * 1.5)),
    )

    for recent, max_tokens in attempts:
        try:
            text = complete_with_continuation(
                [system, *recent, final_question], max_tokens
            )

            if text:
                return text

        except Exception as exc:
            if not is_tool_failure(exc):
                raise

    if results_text:
        return (
            "I found these results but could not write a summary:\n\n"
            + results_text[:1500]
        )

    return "I could not generate an answer. Please rephrase and try again."


def run_agent(question, history, trace=None, deep=False):
    """Run the tool-calling agent. Returns (answer_text, sources).

    deep=True is for research-style questions: several searches run at the
    same time, then one long final answer is written.
    """
    budget = DEEP_BUDGET if deep else SIMPLE_BUDGET

    if trace is None:
        trace = []

    sources = []
    gathered = []   # (tool name, text shown to the model)
    seen_calls = {}

    messages = [
        SystemMessage(content=build_system_prompt()),
        *get_recent_messages(history),
        HumanMessage(content=question),
    ]

    tool_model = get_tool_model()
    rounds_done = 0
    nudged = False

    for _ in range(budget.tool_rounds + 3):

        if rounds_done >= budget.tool_rounds:
            break

        try:
            response = invoke_with_retry(
                tool_model, messages, max_tokens=budget.reply_tokens
            )
        except Exception as exc:
            if is_tool_failure(exc):
                break   # fall back to a plain answer below
            raise

        messages.append(response)

        tool_calls = getattr(response, "tool_calls", []) or []

        if not tool_calls:
            text = content_to_text(response.content).strip()

            # Research questions must search first. Remind the model once.
            if deep and not gathered and not nudged:
                nudged = True
                messages.append(
                    HumanMessage(
                        content="You must call web_search before answering. "
                                "Make one short, specific search per topic "
                                "(up to 5 searches in this step)."
                    )
                )
                continue

            if text and not was_truncated(response):
                return text, sources

            break

        # Decide which calls to run. Every call still needs a matching
        # ToolMessage afterwards, otherwise the API returns an error.
        results_by_index = {}
        planned = []

        for index, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get("name") or "tool"

            if index >= budget.tools_per_round:
                results_by_index[index] = (
                    "Skipped: too many tool calls requested at once."
                )
                continue

            args = sanitize_tool_args(tool_name, tool_call.get("args"))
            call_key = (tool_name, json.dumps(args, sort_keys=True, default=str))

            if call_key in seen_calls:
                results_by_index[index] = (
                    "This exact request was already made. Use the "
                    "earlier results and give your final answer now."
                )
                continue

            seen_calls[call_key] = True
            trace.append(tool_name)
            planned.append((index, tool_name, args))

        # Run the allowed calls at the same time (much faster for searches).
        if planned:
            def execute(item):
                _, name, args = item
                return run_tool_call({"name": name, "args": args})

            with ThreadPoolExecutor(max_workers=len(planned)) as pool:
                raw_results = list(pool.map(execute, planned))

            for (index, name, _), raw_result in zip(planned, raw_results):
                text = prepare_tool_result(name, raw_result, sources)
                results_by_index[index] = text
                gathered.append((name, text))

        for index, tool_call in enumerate(tool_calls):
            messages.append(
                ToolMessage(
                    content=results_by_index.get(index, ""),
                    tool_call_id=tool_call.get("id") or f"call_{index}",
                )
            )

        rounds_done += 1

        # Research mode: all searches ran together, so write the answer now.
        if deep and gathered:
            break

    return generate_final_answer(question, history, gathered, budget), sources


def format_sources(sources, answer):
    """Format sources as clickable Markdown links.

    Lists the sources the answer cites, e.g. [1]. If the answer cites none,
    lists every source that was consulted.
    """
    if not sources:
        return ""

    cited = {int(n) for n in re.findall(r"\[(\d{1,3})\]", answer or "")}
    chosen = [s for s in sources if s["id"] in cited] or sources

    lines = [SOURCES_MARKER, ""]

    for source in chosen:
        title = re.sub(r"\s+", " ", source["title"]).strip()
        title = title.replace("[", "(").replace("]", ")") or "Untitled source"
        url = source["url"].replace(" ", "%20").replace(")", "%29")

        lines.append(f"- **[{source['id']}]** [{title}]({url})")

    return "\n".join(lines)


# ---------------- VERIFIER SEARCH ADAPTER ----------------

def search_json_to_text(result):
    """Turn web_search JSON into plain text; pass other text through."""
    try:
        data = json.loads(result) if isinstance(result, str) else result
    except ValueError:
        return str(result)

    if not isinstance(data, dict) or data.get("type") != "web_search_results":
        return str(result)

    results = data.get("results") or []

    if not results:
        return data.get("error") or "No search results found."

    blocks = []

    for item in results:
        blocks.append(
            f"{item.get('title', 'Untitled')}\n"
            f"{item.get('url', '')}\n"
            f"{item.get('snippet', '')}"
        )

    return "\n\n".join(blocks)


class PlainTextSearch:
    """Wraps the web_search tool so the claim verifier receives plain text
    instead of JSON, whichever way verifier.py calls the tool."""

    def __init__(self, search_tool):
        self._tool = search_tool
        self.name = getattr(search_tool, "name", "web_search")

        raw_func = getattr(search_tool, "func", None)
        if callable(raw_func):
            self.func = lambda *a, **k: search_json_to_text(raw_func(*a, **k))

    def invoke(self, *args, **kwargs):
        return search_json_to_text(self._tool.invoke(*args, **kwargs))

    def run(self, *args, **kwargs):
        return search_json_to_text(self._tool.run(*args, **kwargs))

    def __call__(self, *args, **kwargs):
        if args and not kwargs:
            payload = args[0]
        else:
            payload = kwargs or {}

        return self.invoke(payload)

    def __getattr__(self, item):
        return getattr(self._tool, item)


verifier_search = PlainTextSearch(web_search)


# ---------------- DASHBOARD RENDERING ----------------

ICON_PATHS = {
    "home": '<path d="M3 11l9-8 9 8v10a1 1 0 01-1 1h-5v-6H9v6H4a1 1 0 01-1-1z"/>',
    "bolt": '<path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/>',
    "flow": '<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="12" r="2.5"/>'
            '<circle cx="6" cy="18" r="2.5"/>'
            '<path d="M8.5 6h4a3 3 0 013 3v.5M8.5 18h4a3 3 0 003-3v-.5"/>',
    "plug": '<path d="M9 2v6M15 2v6M6 8h12v4a6 6 0 01-12 0zM12 18v4"/>',
    "file": '<path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z"/>'
            '<path d="M14 3v5h5M9 13h6M9 17h6"/>',
    "gear": '<circle cx="12" cy="12" r="3"/>'
            '<path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9L7 7M17 17l2.1 2.1'
            'M19.1 4.9L17 7M7 17l-2.1 2.1"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.5-4.5"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>'
              '<path d="M9 12l2 2 4-4"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/>'
              '<circle cx="12" cy="12" r="1.5"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "sparkle": '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/>'
            '<path d="M8 11V8a4 4 0 018 0v3"/>',
    "pen": '<path d="M4 20l4-1 11-11-3-3L5 16z"/>',
    "chat": '<path d="M4 5h16v11H9l-5 4z"/><path d="M8 10h8M8 13h5"/>',
    "check": '<path d="M5 12l5 5 9-10"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
}


def icon(name, size=20):
    return (
        f'<svg class="cp-ico" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f'{ICON_PATHS[name]}</svg>'
    )


def now_label():
    return time.strftime("%H:%M:%S")


def new_state():
    return {
        "queries": 0,
        "ok": 0,
        "seconds_saved": 0,
        "tool_calls": 0,
        "cards": {
            "research": ("idle", ""),
            "document": ("idle", ""),
            "verify": ("idle", ""),
        },
        "log": [],
    }


SIDEBAR_HTML = f"""
<div class="cp-brand">
  <svg width="40" height="40" viewBox="0 0 48 48" fill="none" aria-hidden="true">
    <path d="M24 3l17 10v22L24 45 7 35V13z" stroke="#3b82f6" stroke-width="2.4"/>
    <path d="M24 3v20m0 0l17-10M24 23L7 13M24 23v22" stroke="#8b5cf6" stroke-width="1.8"/>
    <path d="M24 14l8 4.6v9.2L24 32.4l-8-4.6v-9.2z" fill="rgba(59,130,246,.25)"
          stroke="#7dd3fc" stroke-width="1.4"/>
  </svg>
  <div>
    <div class="cp-brand-name">AI AGENT</div>
    <div class="cp-brand-sub">RESEARCH COPILOT</div>
  </div>
</div>
<nav class="cp-nav">
  <a class="active" href="#cp-top">{icon("home")}<span>Overview</span></a>
  <a href="#cp-actions">{icon("bolt")}<span>Actions</span></a>
  <a href="#cp-chat-anchor">{icon("flow")}<span>Workflows</span></a>
  <a href="#cp-tools-anchor">{icon("plug")}<span>Integrations</span></a>
  <a href="#cp-activity">{icon("file")}<span>Logs</span></a>
  <a href="#cp-tools-anchor">{icon("gear")}<span>Settings</span></a>
</nav>
<div id="cp-tools-anchor" class="cp-side-title">WORKSPACE TOOLS</div>
"""

EVOLUTION_HTML = f"""
<div class="cp-evo">
  <div class="cp-side-title">AGENT EVOLUTION</div>
  <div class="cp-evo-card">
    <div class="cp-evo-ico">{icon("chat", 22)}</div>
    <div><b>Chatbot</b><small>Read-Only</small></div>
    <span class="cp-evo-end">{icon("lock", 16)}</span>
  </div>
  <div class="cp-evo-link"><i></i><i></i><i></i></div>
  <div class="cp-evo-card active">
    <div class="cp-evo-ico glow">{icon("sparkle", 22)}</div>
    <div><b>AI Agent</b><small class="hl">Search + Verify</small></div>
    <span class="cp-evo-end hl">{icon("pen", 16)}</span>
  </div>
  <p class="cp-evo-tag">From insight to action.<br><span>The power to execute.</span></p>
</div>
"""

BRAIN_HTML = """
<div class="cp-brain-wrap">
  <div class="cp-section-title blue">AI AGENT BRAIN</div>
  <div class="cp-section-sub">Reasoning &amp; Decision Engine</div>
  <svg class="cp-brain" viewBox="0 0 320 340" aria-hidden="true">
    <defs>
      <filter id="cpGlow" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="3.5" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <radialGradient id="cpFill" cx="50%" cy="45%" r="62%">
        <stop offset="0" stop-color="#3b82f6" stop-opacity=".55"/>
        <stop offset="1" stop-color="#0b1a5c" stop-opacity=".12"/>
      </radialGradient>
      <linearGradient id="cpBeam" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="#3b82f6" stop-opacity=".28"/>
        <stop offset="1" stop-color="#3b82f6" stop-opacity="0"/>
      </linearGradient>
      <radialGradient id="cpPed" cx="50%" cy="50%" r="50%">
        <stop offset="0" stop-color="#60a5fa" stop-opacity=".9"/>
        <stop offset="1" stop-color="#1d4ed8" stop-opacity="0"/>
      </radialGradient>
    </defs>

    <ellipse cx="160" cy="300" rx="124" ry="26" fill="url(#cpPed)" opacity=".35"/>
    <ellipse cx="160" cy="298" rx="108" ry="22" fill="none" stroke="#3b82f6" stroke-width="2" filter="url(#cpGlow)"/>
    <ellipse cx="160" cy="294" rx="82" ry="16" fill="rgba(37,99,235,.35)" stroke="#60a5fa" stroke-width="2" filter="url(#cpGlow)"/>
    <ellipse cx="160" cy="290" rx="58" ry="10" fill="rgba(147,197,253,.5)" stroke="#bfdbfe" stroke-width="1.5"/>
    <path d="M112 236L100 290H220L208 236Z" fill="url(#cpBeam)"/>

    <g class="cp-brain-g" filter="url(#cpGlow)" fill="none" stroke="#5aa9ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path fill="url(#cpFill)" stroke-width="2.6" d="M62 150C40 120 52 78 92 66C104 36 150 24 178 44C212 30 258 50 262 92C286 110 288 152 262 172C262 196 236 212 212 204C200 222 172 226 160 210C150 226 124 224 118 206C92 214 66 190 74 166C68 162 64 158 62 150Z"/>
      <path d="M92 66C108 86 96 104 118 114C132 120 140 106 138 92"/>
      <path d="M178 44C170 66 186 78 176 98C170 112 150 112 146 126"/>
      <path d="M262 92C240 96 232 112 214 108C200 106 196 90 204 76"/>
      <path d="M74 166C96 160 100 140 124 144C142 148 140 168 160 168"/>
      <path d="M262 172C238 170 230 150 208 154C190 158 190 180 168 184"/>
      <path d="M118 206C122 190 140 190 150 180"/>
      <path d="M212 204C206 188 190 192 186 204"/>
      <path d="M120 114C104 124 108 144 124 144"/>
      <path d="M214 108C226 122 214 140 208 154"/>
      <path d="M62 150C80 140 90 126 92 110"/>
      <path d="M150 214C148 236 150 250 156 262H172C176 248 174 232 172 216"/>
    </g>
    <g fill="#7dd3fc" class="cp-nodes">
      <circle cx="118" cy="114" r="3"/><circle cx="146" cy="126" r="3"/>
      <circle cx="214" cy="108" r="3"/><circle cx="124" cy="144" r="3"/>
      <circle cx="208" cy="154" r="3"/><circle cx="168" cy="184" r="3"/>
      <circle cx="176" cy="98" r="3"/><circle cx="92" cy="66" r="3"/>
    </g>
    <g stroke="#60a5fa" stroke-width="1.4" opacity=".7" stroke-linecap="round">
      <path d="M10 130h34"/><path d="M4 150h46"/><path d="M14 170h30"/>
      <path d="M22 118h22"/><path d="M30 186h18"/>
      <circle cx="8" cy="130" r="2" fill="#60a5fa"/><circle cx="2" cy="150" r="2" fill="#60a5fa"/>
    </g>
  </svg>
</div>
"""

CONNECTOR_HTML = """
<div class="cp-conn" aria-hidden="true">
  <svg viewBox="0 0 100 400" preserveAspectRatio="none">
    <defs>
      <linearGradient id="cpL1" x1="0" x2="1"><stop offset="0" stop-color="#3b82f6"/><stop offset="1" stop-color="#60a5fa"/></linearGradient>
      <linearGradient id="cpL2" x1="0" x2="1"><stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#a78bfa"/></linearGradient>
      <linearGradient id="cpL3" x1="0" x2="1"><stop offset="0" stop-color="#7c3aed"/><stop offset="1" stop-color="#c084fc"/></linearGradient>
    </defs>
    <path d="M0 212C55 212 45 61 100 61" stroke="url(#cpL1)"/>
    <path d="M0 232C55 232 45 200 100 200" stroke="url(#cpL2)"/>
    <path d="M0 252C55 252 45 339 100 339" stroke="url(#cpL3)"/>
  </svg>
  <i class="dot d1"></i><i class="dot d2"></i><i class="dot d3"></i>
</div>
"""

BANNER_HTML = f"""
<div class="cp-banner">
  <span class="cp-banner-ico">{icon("sparkle", 26)}</span>
  <div>
    <b>Research Copilot is actively searching, reading and verifying for you.</b>
    <span>Autonomous. Reliable. Impactful.</span>
  </div>
</div>
"""

FOOTER_HTML = """
<div id="copilot-footer">
  Research Copilot AI · Powered by Python, Gradio and Groq
</div>
"""

CARD_DEFS = [
    ("research", "Web Research", "Searching the web and running tools", "search", "blue"),
    ("document", "Document Analysis", "Reading your uploaded file", "file", "purple"),
    ("verify", "Claim Verification", "Fact-checking the final answer", "shield", "violet"),
]

STATUS_META = {
    "idle": ("Waiting", 0),
    "progress": ("In Progress", 60),
    "done": ("Completed", 100),
    "skipped": ("Skipped", 0),
    "error": ("Failed", 100),
}


def render_header(working):
    status = "Working" if working else "Active"
    cls = "working" if working else "active"
    subtitle = (
        "Autonomous Actions in Progress" if working else "Ready for your next task"
    )
    return f"""
<div id="cp-top" class="cp-header">
  <div>
    <h1>RESEARCH COPILOT CONTROL PANEL</h1>
    <p>{subtitle} <i class="cp-live {cls}"></i></p>
  </div>
  <div class="cp-header-right">
    <div class="cp-status-box">
      <span class="cp-status-dot {cls}"></span>
      <div><small>Agent Status</small><b>{status}</b></div>
    </div>
    <a class="cp-log-btn" href="#cp-activity">{icon("list", 18)}<span>View Activity Log</span></a>
  </div>
</div>
"""


def render_cards(cards):
    out = ['<div id="cp-actions" class="cp-section-title violet">ACTIONS IN PROGRESS</div>',
           '<div class="cp-cards">']

    for key, title, desc, ico, color in CARD_DEFS:
        status, stamp = cards.get(key, ("idle", ""))
        label, pct = STATUS_META.get(status, STATUS_META["idle"])

        if status == "done":
            mark = f'<span class="cp-mark">{icon("check", 12)}</span>'
        elif status == "progress":
            mark = '<span class="cp-spin"></span>'
        else:
            mark = '<span class="cp-mark empty"></span>'

        out.append(f"""
<div class="cp-card {color} is-{status}">
  <div class="cp-card-top">
    <div class="cp-card-icon">{icon(ico, 26)}</div>
    <div>
      <div class="cp-card-title">{title}</div>
      <div class="cp-card-desc">{desc}</div>
    </div>
  </div>
  <div class="cp-bar-row">
    <div class="cp-bar"><span style="width:{pct}%"></span></div>
    <b>{pct}%</b>
  </div>
  <div class="cp-card-foot">
    <span class="cp-st">{mark}{label}</span>
    <span>{stamp}</span>
  </div>
</div>""")

    out.append("</div>")
    return "".join(out)


def format_saved(seconds):
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds / 3600:.1f}h"


def render_stats(state):
    queries = state["queries"]
    rate = f"{state['ok'] / queries * 100:.1f}%" if queries else "—"
    rate_note = "High Reliability" if queries and state["ok"] / queries >= 0.9 else (
        "Needs attention" if queries else "No tasks yet"
    )

    items = [
        ("bolt", "Tasks Today", str(queries), "Questions handled", "blue"),
        ("target", "Success Rate", rate, rate_note, "violet"),
        ("clock", "Time Saved", format_saved(state["seconds_saved"]), "vs manual research", "blue"),
        ("layers", "Tool Calls", str(state["tool_calls"]), "Searches and calculations", "violet"),
    ]

    cells = "".join(f"""
<div class="cp-stat">
  <div class="cp-stat-ico {color}">{icon(ico, 22)}</div>
  <div>
    <small>{label}</small>
    <b>{value}</b>
    <em>{note}</em>
  </div>
</div>""" for ico, label, value, note, color in items)

    return f'<div class="cp-stats">{cells}</div>'


def render_activity(state):
    entries = state["log"]

    if not entries:
        rows = '<div class="cp-log-empty">No activity yet. Ask a question to get started.</div>'
    else:
        rows = "".join(f"""
<div class="cp-log-row">
  <span class="cp-log-ico {kind}">{icon(ico, 16)}</span>
  <span class="cp-log-text">{html.escape(text)}</span>
  <span class="cp-log-time">{stamp}</span>
</div>""" for kind, ico, text, stamp in reversed(entries))

    return f"""
<div id="cp-activity" class="cp-activity">
  <div class="cp-activity-head">Recent Activity</div>
  {rows}
</div>
"""


def dashboard(state, working=False):
    return (
        render_header(working),
        render_cards(state["cards"]),
        render_stats(state),
        render_activity(state),
    )


def add_log(state, kind, ico, text):
    state["log"].append((kind, ico, text, now_label()))
    state["log"] = state["log"][-MAX_LOG_ENTRIES:]


# ---------------- CHAT FUNCTION ----------------

def respond(user_message, history, uploaded_file, verify_claims, state):
    history = normalize_history(history)
    state = state or new_state()
    question = (user_message or "").strip()

    if not question and not uploaded_file:
        yield (history, "", state, *dashboard(state))
        return

    if not question:
        question = "Summarize the uploaded document and explain its key points."

    question = question[:MAX_QUESTION_CHARS]
    has_file = bool(uploaded_file)

    # Show the user's message and the live "in progress" cards right away.
    pending = history + [{"role": "user", "content": question}]
    state["cards"] = {
        "research": ("progress", now_label()),
        "document": ("progress", now_label()) if has_file else ("skipped", now_label()),
        "verify": ("idle", ""),
    }
    yield (pending, "", state, *dashboard(state, working=True))

    # Research-style questions get more searches and a longer answer.
    deep = is_complex_request(question)

    # What the model sees (may include document text).
    agent_question = question
    answer_ok = False
    trace = []
    sources = []

    try:
        if has_file:
            file_path = getattr(uploaded_file, "name", uploaded_file)
            document_text = read_uploaded_file(str(file_path))
            agent_question += (
                "\n\nUploaded document text:\n"
                + document_text[:MAX_DOCUMENT_CHARS]
            )

        answer, sources = run_agent(agent_question, history, trace, deep=deep)

        if not answer or not str(answer).strip():
            answer = "No response was generated. Please try again."
        else:
            answer_ok = True

    except Exception as exc:
        print("AGENT ERROR:", repr(exc))
        answer = friendly_error(exc)

    answer = str(answer)

    # Update the cards and log with what actually happened.
    stamp = now_label()
    research_status = "done" if answer_ok else "error"
    doc_status = ("done" if answer_ok else "error") if has_file else "skipped"
    state["cards"] = {
        "research": (research_status, stamp),
        "document": (doc_status, stamp),
        "verify": ("idle", ""),
    }

    state["queries"] += 1
    state["tool_calls"] += len(trace)

    if answer_ok:
        state["ok"] += 1
        state["seconds_saved"] += SECONDS_SAVED_PER_ANSWER
        add_log(state, "green", "check", f"Answered: {question[:48]}")
        if deep:
            add_log(state, "blue", "layers", "Deep research mode: parallel searches")
        for name in dict.fromkeys(trace):
            add_log(state, "blue", "search", f"Tool used: {name}")
        if has_file:
            add_log(state, "violet", "file", "Document analyzed")
    else:
        add_log(state, "red", "bolt", "Request failed. Check the chat for details.")

    will_verify = bool(verify_claims) and answer_ok and not has_file

    if will_verify:
        state["cards"]["verify"] = ("progress", now_label())

    interim = pending + [{"role": "assistant", "content": answer}]
    yield (interim, "", state, *dashboard(state, working=will_verify))

    # Claim Verifier: fact-check the answer with fresh web searches.
    # Skipped for document questions, because the claims come from the
    # user's file and may not exist on the web.
    if will_verify:
        report = None

        try:
            report = verify_answer(answer, get_base_model(), verifier_search)
        except Exception as exc:
            print("VERIFIER ERROR:", repr(exc))

        if report:
            answer = f"{answer}\n\n{report}"
            add_log(state, "violet", "shield", "Claims verified")

        state["cards"]["verify"] = ("done" if report else "skipped", now_label())
    else:
        state["cards"]["verify"] = ("skipped", now_label())

    # Add clickable source references to the final answer.
    if answer_ok:
        try:
            answer += format_sources(sources, answer)
        except Exception as exc:
            print("SOURCES ERROR:", repr(exc))

    # Messages format: list of {"role", "content"} dictionaries.
    final_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    yield (final_history, "", state, *dashboard(state))


# ---------------- GRADIO INTERFACE ----------------

# Gradio versions differ on where `css` and Chatbot `type` belong,
# so check what the installed version supports.
blocks_kwargs = {"title": "Research Copilot AI"}
launch_kwargs = {}

if "css" in inspect.signature(gr.Blocks.__init__).parameters:
    blocks_kwargs["css"] = CUSTOM_CSS
else:
    launch_kwargs["css"] = CUSTOM_CSS

# This project uses dictionary-based chat history.
chatbot_kwargs = {
    "label": "Conversation",
    "show_label": False,
    "height": 470,
    "elem_id": "copilot-chatbot",
}

if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
    chatbot_kwargs["type"] = "messages"

_initial_state = new_state()

with gr.Blocks(**blocks_kwargs) as demo:  # type: ignore

    app_state = gr.State(new_state())

    with gr.Column(elem_id="cp-page"):

        # Main container
        with gr.Row(elem_id="cp-shell", equal_height=False):

            # Sidebar
            with gr.Column(
                scale=0,
                min_width=250,
                elem_id="cp-sidebar",
            ):
                gr.HTML(SIDEBAR_HTML, elem_classes=["cp-flat"])

                file_upload = gr.File(
                    label="Upload a document",
                    file_count="single",
                    elem_id="copilot-upload",
                )

                verify_toggle = gr.Checkbox(
                    label="Verify claims after answering",
                    value=True,
                    info="Fact-checks web answers with fresh searches (a bit slower).",
                    elem_id="copilot-verify",
                )

                gr.HTML(EVOLUTION_HTML, elem_classes=["cp-flat"])

            # Main dashboard
            with gr.Column(
                scale=1,
                min_width=320,
                elem_id="cp-main",
            ):
                header_html = gr.HTML(
                    render_header(False),
                    elem_classes=["cp-flat"],
                )

                with gr.Column(elem_id="cp-center"):

                    with gr.Row(elem_id="cp-stage-row", equal_height=False):
                        gr.HTML(
                            f'<div class="cp-stage">{BRAIN_HTML}{CONNECTOR_HTML}</div>',
                            elem_classes=["cp-flat", "cp-stage-wrap"],
                        )

                        cards_html = gr.HTML(
                            render_cards(_initial_state["cards"]),
                            elem_classes=["cp-flat", "cp-cards-wrap"],
                        )

                    stats_html = gr.HTML(
                        render_stats(_initial_state),
                        elem_classes=["cp-flat"],
                    )

                    gr.HTML(BANNER_HTML, elem_classes=["cp-flat"])

                    activity_html = gr.HTML(
                        render_activity(_initial_state),
                        elem_classes=["cp-flat"],
                    )

        # Research Workspace: same outer container, full width
        with gr.Column(elem_id="cp-chat-panel"):

            gr.HTML(
                """
                <div id="cp-chat-anchor" class="cp-chat-head">
                    <div>
                        <div class="cp-section-title blue">RESEARCH WORKSPACE</div>
                        <div class="cp-section-sub">
                            Ask questions, explore topics and analyze documents.
                        </div>
                        <div class="cp-connected">
                            <i></i> Connected: Groq
                        </div>
                    </div>
                    <div class="cp-cloud">GROQ</div>
                </div>
                """,
                elem_classes=["cp-flat"],
            )

            chatbot = gr.Chatbot(**chatbot_kwargs)

            user_message = gr.Textbox(
                placeholder="Ask a question or explore a research topic...",
                label="Message",
                show_label=False,
                lines=2,
                elem_id="copilot-prompt",
            )

            with gr.Row():
                send_button = gr.Button(
                    "Send message",
                    variant="primary",
                    elem_id="copilot-send",
                )

                clear_button = gr.Button(
                    "Clear conversation",
                    elem_id="copilot-clear",
                )

        gr.HTML(FOOTER_HTML, elem_classes=["cp-flat"])

    # ==========================================
    # EVENT HANDLING
    # ==========================================
    outputs = [
        chatbot,
        user_message,
        app_state,
        header_html,
        cards_html,
        stats_html,
        activity_html,
    ]

    inputs = [
        user_message,
        chatbot,
        file_upload,
        verify_toggle,
        app_state,
    ]

    user_message.submit(fn=respond, inputs=inputs, outputs=outputs)
    send_button.click(fn=respond, inputs=inputs, outputs=outputs)

    clear_button.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, user_message],
    )


if __name__ == "__main__":
    demo.launch(**launch_kwargs)