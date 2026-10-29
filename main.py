"""Multi-tool Autonomous AI Research Agent with Claim Verifier."""

import inspect
import os
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
- Do not invent sources or facts.
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


def run_tool_call(tool_call):
    """Execute a single tool call and return the result text."""
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
        return str(result)[:MAX_TOOL_RESULT_CHARS]

    except Exception as exc:
        return f"Tool failed: {type(exc).__name__}: {exc}"


def run_agent(question, history):
    """Run a short tool-calling loop without sending unlimited history."""

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
            return content_to_text(response.content)

        # Every tool call needs a matching ToolMessage, otherwise the API
        # returns an error. Only the first few are actually executed.
        for index, tool_call in enumerate(tool_calls):
            if index < MAX_TOOLS_PER_ROUND:
                tool_result = run_tool_call(tool_call)
            else:
                tool_result = "Skipped: too many tool calls requested at once."

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call.get("id"),
                )
            )

    # Tool rounds used up: ask for a final answer using a model without tools.
    messages.append(
        HumanMessage(
            content="Using the available results, give a concise final answer. "
                    "Do not call any more tools."
        )
    )

    final_response = get_base_model().invoke(messages)

    return content_to_text(final_response.content)


# ---------------- CHAT FUNCTION ----------------

def respond(user_message, history, uploaded_file, verify_claims):
    history = normalize_history(history)
    question = (user_message or "").strip()

    if not question and not uploaded_file:
        return history, ""

    if not question:
        question = "Summarize the uploaded document and explain its key points."

    question = question[:MAX_QUESTION_CHARS]

    # What the model sees (may include document text).
    agent_question = question
    answer_ok = False

    try:
        if uploaded_file:
            file_path = getattr(uploaded_file, "name", uploaded_file)
            document_text = read_uploaded_file(str(file_path))
            agent_question += (
                "\n\nUploaded document text:\n"
                + document_text[:MAX_DOCUMENT_CHARS]
            )

        answer = run_agent(agent_question, history)

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

    # Claim Verifier: fact-check the answer with fresh web searches.
    # Skipped for document questions, because the claims come from the
    # user's file and may not exist on the web.
    if verify_claims and answer_ok and not uploaded_file:
        report = verify_answer(answer, get_base_model(), web_search)

        if report:
            answer = f"{answer}\n\n{report}"

    # Messages format: list of {"role", "content"} dictionaries.
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})

    return history, ""


# ---------------- GRADIO INTERFACE ----------------

# Gradio versions differ on where `css` and Chatbot `type` belong,
# so check what the installed version supports.
blocks_kwargs = {"title": "Research Copilot AI"}
launch_kwargs = {}

if "css" in inspect.signature(gr.Blocks.__init__).parameters:
    blocks_kwargs["css"] = CUSTOM_CSS
else:
    launch_kwargs["css"] = CUSTOM_CSS

chatbot_kwargs = {
    "label": "Conversation",
    "height": 500,
    "elem_id": "copilot-chatbot",
}

if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
    chatbot_kwargs["type"] = "messages"


with gr.Blocks(**blocks_kwargs) as demo:

    # Header
    gr.HTML("""
    <div id="copilot-header">
        <h1>Research Copilot AI</h1>
        <p>
            Your intelligent workspace for research, documents,
            web search, calculations, and AI-powered answers
            that fact-check themselves.
        </p>
    </div>
    """)

    with gr.Row(elem_id="copilot-workspace"):

        # Left sidebar
        with gr.Column(scale=1, min_width=250, elem_id="copilot-sidebar"):
            gr.Markdown("### Research Workspace")
            gr.Markdown(
                "Upload a document to ask questions about its contents."
            )

            file_upload = gr.File(
                label="Upload a document",
                file_count="single",
                elem_id="copilot-upload",
            )

            verify_toggle = gr.Checkbox(
                label="Verify claims after answering",
                value=True,
                info="Fact-checks the answer with fresh web searches "
                     "(web questions only, a bit slower).",
                elem_id="copilot-verify",
            )

            gr.Markdown("### Available capabilities")
            gr.Markdown(
                "- Web search\n"
                "- Claim verification\n"
                "- Safe calculations\n"
                "- PDF, TXT and DOCX reading\n"
                "- AI research assistance"
            )

            gr.Markdown(
                "*Tip: Ask one clear question at a time for focused results.*"
            )

        # Main chat panel
        with gr.Column(scale=3, min_width=320, elem_id="copilot-chat-panel"):
            gr.Markdown("### Ask your research copilot")

            chatbot = gr.Chatbot(**chatbot_kwargs)

            with gr.Row():
                user_message = gr.Textbox(
                    placeholder="Ask a question or explore a research topic...",
                    label="",
                    lines=2,
                    scale=5,
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

    gr.HTML("""
    <div id="copilot-footer">
        Research Copilot AI · Powered by Python, Gradio and Groq
    </div>
    """)

    # Submit using either Enter or the Send button.
    user_message.submit(
        fn=respond,
        inputs=[user_message, chatbot, file_upload, verify_toggle],
        outputs=[chatbot, user_message],
    )

    send_button.click(
        fn=respond,
        inputs=[user_message, chatbot, file_upload, verify_toggle],
        outputs=[chatbot, user_message],
    )

    clear_button.click(
        fn=lambda: ([], ""),
        inputs=[],
        outputs=[chatbot, user_message],
    )


if __name__ == "__main__":
    demo.launch(**launch_kwargs)