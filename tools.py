"""Tools for the Multi-tool Autonomous AI Research Agent.

Provides:
- web_search: resilient web search (tries several search back ends, retries,
  caches repeated queries) returning structured JSON.
- calculator: safe arithmetic without eval().
- read_uploaded_file: TXT / MD / CSV / PDF / DOCX text extraction.
"""
import ast
import json
import operator
import re
import time
from pathlib import Path

from langchain_core.tools import tool


# ---------------- CONFIGURATION ----------------

MAX_SEARCH_RESULTS = 5
MAX_SNIPPET_CHARS = 300
MAX_TITLE_CHARS = 200
SEARCH_ATTEMPTS = 2
SEARCH_TIMEOUT = 10
SEARCH_RETRY_DELAY = 1.2
SEARCH_CACHE_SECONDS = 300

MAX_FILE_CHARS = 3000
MAX_PDF_PAGES = 10
MAX_FILE_BYTES = 15 * 1024 * 1024

_search_cache = {}


# ---------------- WEB SEARCH ----------------

def _clean_url(value):
    url = str(value or "").strip()
    return url if url.startswith(("https://", "http://")) else ""


def _normalize_results(raw_results):
    """Convert results from any search back end into one clean format."""
    cleaned = []
    seen = set()

    for item in raw_results or []:
        if not isinstance(item, dict):
            continue

        url = _clean_url(
            item.get("href") or item.get("link") or item.get("url")
        )

        if not url or url in seen:
            continue

        seen.add(url)

        title = str(item.get("title") or "Untitled source").strip()
        snippet = str(
            item.get("body") or item.get("snippet") or ""
        ).strip()

        cleaned.append({
            "id": len(cleaned) + 1,
            "title": re.sub(r"\s+", " ", title)[:MAX_TITLE_CHARS],
            "url": url,
            "snippet": re.sub(r"\s+", " ", snippet)[:MAX_SNIPPET_CHARS],
        })

        if len(cleaned) >= MAX_SEARCH_RESULTS:
            break

    return cleaned


def _make_ddgs(ddgs_class):
    """Create a search client with a timeout, if this version supports it."""
    try:
        return ddgs_class(timeout=SEARCH_TIMEOUT)
    except TypeError:
        return ddgs_class()


def _search_with_ddgs(query):
    """Newest package name: `pip install ddgs`."""
    from ddgs import DDGS

    return _make_ddgs(DDGS).text(query, max_results=MAX_SEARCH_RESULTS)


def _search_with_duckduckgo_search(query):
    """Older package name: `pip install duckduckgo-search`."""
    from duckduckgo_search import DDGS

    return _make_ddgs(DDGS).text(query, max_results=MAX_SEARCH_RESULTS)


def _search_with_langchain(query):
    """LangChain wrapper (needs one of the packages above installed)."""
    from langchain_community.tools import DuckDuckGoSearchResults

    engine = DuckDuckGoSearchResults(
        output_format="list",
        max_results=MAX_SEARCH_RESULTS,
    )
    return engine.invoke(query)


_BACKENDS = (
    _search_with_ddgs,
    _search_with_duckduckgo_search,
    _search_with_langchain,
)


def _run_search(query):
    """Try each back end, with retries. Returns (results, error_text)."""
    errors = []

    for attempt in range(SEARCH_ATTEMPTS):
        for backend in _BACKENDS:
            try:
                results = _normalize_results(backend(query))

                if results:
                    return results, ""

            except ImportError:
                continue

            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")

        if attempt < SEARCH_ATTEMPTS - 1:
            time.sleep(SEARCH_RETRY_DELAY)

    if errors:
        return [], errors[-1][:200]

    return [], ""


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Takes one argument, query
    (a short search phrase). Returns source titles, URLs and snippets."""
    query = str(query or "").strip()

    if not query:
        return json.dumps({
            "type": "web_search_results",
            "query": "",
            "results": [],
            "error": "The search query was empty.",
        })

    cache_key = query.lower()
    cached = _search_cache.get(cache_key)

    if cached and time.time() - cached[0] < SEARCH_CACHE_SECONDS:
        return cached[1]

    try:
        results, error = _run_search(query)

    except Exception as exc:
        results, error = [], f"{type(exc).__name__}: {exc}"

    payload = {
        "type": "web_search_results",
        "query": query,
        "results": results,
    }

    if not results:
        payload["error"] = error or "No search results found."

    output = json.dumps(payload, ensure_ascii=False)

    # Only cache successful searches, so failures are retried next time.
    if results:
        _search_cache[cache_key] = (time.time(), output)

    return output


# ---------------- CALCULATOR ----------------

# Safe basic calculator: does not use eval().
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_NUMBER = 10 ** 15
_MAX_EXPRESSION_CHARS = 200


def _calculate(node):
    if isinstance(node, ast.Expression):
        return _calculate(node.body)

    if isinstance(node, ast.Constant):
        # bool is a subclass of int, so reject it explicitly.
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            raise ValueError("Only numbers are allowed.")

        if abs(node.value) > _MAX_NUMBER:
            raise ValueError("Number is too large.")

        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported arithmetic operator.")

        left = _calculate(node.left)
        right = _calculate(node.right)

        if op_type is ast.Pow and (abs(right) > 10 or abs(left) > 10 ** 6):
            raise ValueError("Exponent or base is too large.")

        result = _ALLOWED_OPERATORS[op_type](left, right)

        if isinstance(result, complex):
            raise ValueError("The result is not a real number.")

        if isinstance(result, (int, float)) and abs(result) > _MAX_NUMBER ** 2:
            raise ValueError("Result is too large.")

        return result

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_UNARY:
            raise ValueError("Unsupported unary operator.")

        return _ALLOWED_UNARY[op_type](_calculate(node.operand))

    raise ValueError("Only basic arithmetic expressions are allowed.")


def _tidy_expression(expression):
    """Accept the symbols people and models commonly write."""
    text = str(expression or "").strip()
    text = text.replace(",", "")
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-")
    text = text.replace("^", "**")
    return text


@tool
def calculator(expression: str) -> str:
    """Calculate basic arithmetic. Takes one argument, expression,
    for example (25 + 5) / 2."""
    try:
        text = _tidy_expression(expression)

        if not text:
            return "Calculation failed: the expression was empty."

        if len(text) > _MAX_EXPRESSION_CHARS:
            return "Calculation failed: the expression is too long."

        tree = ast.parse(text, mode="eval")
        result = _calculate(tree)

        if isinstance(result, float):
            result = round(result, 10)

        return str(result)

    except ZeroDivisionError:
        return "Calculation failed: division by zero."

    except Exception as exc:
        return f"Calculation failed: {exc}"


# ---------------- FILE READING ----------------

_TEXT_SUFFIXES = {".txt", ".md", ".csv"}


def read_uploaded_file(file_path) -> str:
    """Read a TXT, MD, CSV, PDF, or DOCX file and return limited text."""

    if not file_path:
        return "No file was provided."

    # Gradio may pass a path string or an object with a .name path.
    path = Path(str(getattr(file_path, "name", file_path)))

    if not path.is_file():
        return "The uploaded file could not be found."

    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return "The file is too large. Please upload a file under 15 MB."

        suffix = path.suffix.lower()

        if suffix in _TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")

        elif suffix == ".pdf":
            text = _read_pdf(path)

        elif suffix == ".docx":
            text = _read_docx(path)

        else:
            return "Unsupported file type. Upload TXT, PDF, or DOCX."

        text = text.strip()

        if not text:
            if suffix == ".pdf":
                return (
                    "No readable text was found. The PDF may be a scanned "
                    "image, which this app cannot read."
                )
            return "No readable text was found in the file."

        if len(text) > MAX_FILE_CHARS:
            text = (
                text[:MAX_FILE_CHARS]
                + "\n\n[Document text shortened to reduce token usage.]"
            )

        return text

    except ImportError as exc:
        missing = "pypdf" if suffix == ".pdf" else "python-docx"
        return (
            f"File reading failed: the '{missing}' package is not installed "
            f"({exc}). Install it with: pip install {missing}"
        )

    except Exception as exc:
        return f"File reading failed: {type(exc).__name__}: {exc}"


def _read_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))

    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("The PDF is password protected.")

    parts = []

    for page in reader.pages[:MAX_PDF_PAGES]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue

    return "\n".join(parts)


def _read_docx(path):
    from docx import Document

    document = Document(str(path))
    parts = [paragraph.text for paragraph in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cells))

    return "\n".join(parts)


tools = [web_search, calculator]