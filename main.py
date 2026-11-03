"""Multi-tool Autonomous AI Research Agent with Claim Verifier.

Control-panel UI: sidebar, live agent-brain dashboard, action cards,
stats row, activity log and the chat workspace.
"""
import json
import html
import inspect
import os
import time
from pathlib import Path

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
CUSTOM_CSS = (BASE_DIR / "style.css").read_text(encoding="utf-8")

# ---------------- CONFIGURATION ----------------

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

MAX_QUESTION_CHARS = 1500
MAX_DOCUMENT_CHARS = 2500
MAX_TOOL_RESULT_CHARS = 1200
MAX_TOOL_ROUNDS = 2
MAX_TOOLS_PER_ROUND = 2
MAX_REPLY_TOKENS = 700
MAX_HISTORY_MESSAGES = 4
MAX_HISTORY_CHARS = 700

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
                "GROQ_API_KEY is missing. Set it in your environment."
            )

        _base_model = ChatGroq(
            model=MODEL_NAME,
            temperature=0.2,
            max_tokens=MAX_REPLY_TOKENS,
            api_key=api_key,
        )

    return _base_model


def get_tool_model():
    """Model with tools bound (used inside the agent loop)."""
    global _tool_model

    if _tool_model is None:
        _tool_model = get_base_model().bind_tools(tools)

    return _tool_model


# ---------------- HELPER FUNCTIONS ----------------

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
    """Remove the verification report so it is not sent back to the model."""
    marker = "\n\n---\n### 🔍 Claim Verification"
    index = text.find(marker)
    return text[:index] if index != -1 else text


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

def get_tool_model():
    """Model with tools bound for agent execution."""
    global _tool_model

    if _tool_model is None:
        _tool_model = get_base_model().bind_tools(
            tools,
            tool_choice="auto",
        )

    return _tool_model


def prepare_tool_result(tool_name, result, sources):
    """Assign consistent reference IDs to web search results."""
    if tool_name != "web_search":
        return result

    try:
        data = json.loads(result)

        if data.get("type") != "web_search_results":
            return result

        for item in data.get("results", []):
            url = item.get("url", "").strip()

            if not url.startswith(("https://", "http://")):
                continue

            # Reuse an existing reference for duplicate URLs.
            existing = next(
                (source for source in sources if source["url"] == url),
                None,
            )

            if existing:
                item["id"] = existing["id"]
            else:
                source_id = len(sources) + 1

                source = {
                    "id": source_id,
                    "title": item.get("title", "Untitled source"),
                    "url": url,
                }

                sources.append(source)
                item["id"] = source_id

        return json.dumps(data, ensure_ascii=False)

    except (ValueError, TypeError, AttributeError):
        return result

def run_agent(question, history, trace=None):
    """Run the tool-calling agent and collect web sources."""

    if trace is None:
        trace = []

    sources = []

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *get_recent_messages(history),
        HumanMessage(content=question),
    ]

    for _ in range(MAX_TOOL_ROUNDS + 1):

        response = get_tool_model().invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", []) or []

        if not tool_calls:
            return content_to_text(response.content), sources

        for index, tool_call in enumerate(tool_calls):

            tool_name = tool_call.get("name") or "tool"

            if index < MAX_TOOLS_PER_ROUND:

                trace.append(tool_name)

                tool_result = run_tool_call(tool_call)

                tool_result = prepare_tool_result(
                    tool_name,
                    tool_result,
                    sources,
                )

            else:
                tool_result = "Skipped: too many tool calls requested at once."

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call.get("id"),
                )
            )

    # Generate the final response without allowing additional tool calls.
    messages.append(
        HumanMessage(
            content=(
                "Now provide your final answer using the available tool results. "
                "Do not request or call any more tools. "
                "Follow the source citation rules."
            )
        )
    )

    final_response = get_base_model().invoke(messages)

    return content_to_text(final_response.content), sources

def format_sources(sources):
    """Format collected sources as clickable Markdown links."""
    if not sources:
        return ""

    lines = [
        "",
        "---",
        "### 🔗 Sources & References",
        "",
    ]

    for source in sources:
        source_id = source["id"]
        title = html.escape(source["title"])
        url = source["url"]

        lines.append(
            f"[{source_id}] [{title}]({url})"
        )

    return "\n".join(lines)
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
        label, pct = STATUS_META[status]

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
        result = selected_tool.invoke(tool_args)
        result = str(result)

        # Preserve structured web results.
        if tool_name == "web_search":
            return result

        return result[:MAX_TOOL_RESULT_CHARS]

    except Exception as exc:
        return f"Tool failed: {type(exc).__name__}: {exc}"

def add_log(state, kind, ico, text):
    state["log"].append((kind, ico, text, now_label()))
    state["log"] = state["log"][-MAX_LOG_ENTRIES:]


# ---------------- CHAT FUNCTION ----------------

def respond(user_message, history, uploaded_file, verify_claims, state):
    trace = []
    sources = []
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

    # What the model sees (may include document text).
    agent_question = question
    answer_ok = False
    trace = []

    try:
        if has_file:
            file_path = getattr(uploaded_file, "name", uploaded_file)
            document_text = read_uploaded_file(str(file_path))
            agent_question += (
                "\n\nUploaded document text:\n"
                + document_text[:MAX_DOCUMENT_CHARS]
            )

        answer ,sources= run_agent(agent_question, history, trace)

        if not answer or not str(answer).strip():
            answer = "No response was generated. Please try again."
        else:
            answer_ok = True

    except Exception as exc:
        print("AGENT ERROR:", repr(exc))
        error_text = str(exc)

        if (
            "413" in error_text
            or "tokens per minute" in error_text.lower()
            or "rate_limit_exceeded" in error_text.lower()
        ):
            answer = (
                "Groq's token limit was exceeded. "
                "Please shorten your question or document."
            )
        else:
            answer = f"An error occurred: {error_text}"

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
            report = verify_answer(answer, get_base_model(), web_search)
        except Exception as exc:
            print("VERIFIER ERROR:", repr(exc))

        if report:
            answer = f"{answer}\n\n{report}"
            add_log(state, "violet", "shield", "Claims verified")

        state["cards"]["verify"] = ("done" if report else "skipped", now_label())
    else:
        state["cards"]["verify"] = ("skipped", now_label())

    # Add clickable source references to the final answer.
    answer += format_sources(sources)
    # Messages format: list of {"role", "content"} dictionaries.
    final_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    yield (final_history, "", state, *dashboard(state))


# ---------------- GRADIO INTERFACE ----------------

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

with gr.Blocks(**blocks_kwargs) as demo: # type: ignore

    app_state = gr.State(new_state())

    with gr.Column(elem_id="cp-page"):

        # Existing main container
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

    # Keep your existing event handling here.
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

    user_message.submit(
        fn=respond,
        inputs=inputs,
        outputs=outputs,
    )

    send_button.click(
        fn=respond,
        inputs=inputs,
        outputs=outputs,
    )

    clear_button.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, user_message],
    )



if __name__ == "__main__":
    demo.launch(**launch_kwargs)
