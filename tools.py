"""Tools for the Multi-tool Autonomous AI Research Agent."""

import ast
import operator
from pathlib import Path

from langchain_core.tools import tool


# Keep tool results small to reduce Groq token usage.
MAX_SEARCH_CHARS = 1500
MAX_FILE_CHARS = 3000
MAX_PDF_PAGES = 10

_search_engine = None


def _get_search_engine():
    """Create the search engine on first use so a missing package
    does not crash the whole app at startup."""
    global _search_engine

    if _search_engine is None:
        from langchain_community.tools import DuckDuckGoSearchRun

        _search_engine = DuckDuckGoSearchRun()

    return _search_engine


@tool
def web_search(query: str) -> str:
    """Search the web for information relevant to a research question."""
    try:
        result = _get_search_engine().run(query)

        if not result:
            return "No search results found."

        return str(result)[:MAX_SEARCH_CHARS]

    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"


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

        if op_type is ast.Pow and abs(right) > 10:
            raise ValueError("Exponent is too large.")

        result = _ALLOWED_OPERATORS[op_type](left, right)

        if isinstance(result, (int, float)) and abs(result) > _MAX_NUMBER ** 2:
            raise ValueError("Result is too large.")

        return result

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)

        if op_type not in _ALLOWED_UNARY:
            raise ValueError("Unsupported unary operator.")

        return _ALLOWED_UNARY[op_type](_calculate(node.operand))

    raise ValueError("Only basic arithmetic expressions are allowed.")


@tool
def calculator(expression: str) -> str:
    """Calculate basic arithmetic, for example (25 + 5) / 2."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _calculate(tree)
        return str(result)

    except ZeroDivisionError:
        return "Calculation failed: division by zero."

    except Exception as exc:
        return f"Calculation failed: {exc}"


def read_uploaded_file(file_path) -> str:
    """Read a TXT, PDF, or DOCX file and return limited extracted text."""

    if not file_path:
        return "No file was provided."

    # Gradio may pass a path string or an object with a .name path.
    path = Path(str(getattr(file_path, "name", file_path)))

    if not path.is_file():
        return "The uploaded file could not be found."

    try:
        suffix = path.suffix.lower()

        if suffix == ".txt":
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

        elif suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            parts = []

            for page in reader.pages[:MAX_PDF_PAGES]:
                parts.append(page.extract_text() or "")

            text = "\n".join(parts)

        elif suffix == ".docx":
            from docx import Document

            document = Document(str(path))
            text = "\n".join(
                paragraph.text for paragraph in document.paragraphs
            )

        else:
            return "Unsupported file type. Upload TXT, PDF, or DOCX."

        text = text.strip()

        if not text:
            return "No readable text was found in the file."

        if len(text) > MAX_FILE_CHARS:
            text = (
                text[:MAX_FILE_CHARS]
                + "\n\n[Document text shortened to reduce token usage.]"
            )

        return text

    except Exception as exc:
        return f"File reading failed: {type(exc).__name__}: {exc}"


tools = [web_search, calculator]