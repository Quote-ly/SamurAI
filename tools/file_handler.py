"""File handling — parse inbound documents and create edited versions."""

import io
import logging
import os
import tempfile

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Store uploaded file content per conversation for editing
# Key: conversation_id, Value: {filename, content_bytes, file_type, text_content}
_uploaded_files: dict[str, dict] = {}

# Store pending edited files for FileConsentCard upload
# Key: conversation_id, Value: {filename, content_bytes, summary}
_pending_edited_files: dict[str, dict] = {}


# ── File Parsing ──────────────────────────────────────────────────────


def parse_docx(content_bytes: bytes) -> str:
    """Extract text from a .docx file."""
    from docx import Document

    doc = Document(io.BytesIO(content_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    # Also extract tables
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            paragraphs.append(" | ".join(cells))

    return "\n".join(paragraphs)


def parse_xlsx(content_bytes: bytes) -> str:
    """Extract text from an .xlsx file."""
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(content_bytes), read_only=True, data_only=True)
    lines = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        lines.append(f"**Sheet: {sheet_name}**")
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(cells):
                lines.append(" | ".join(cells))
        lines.append("")

    wb.close()
    return "\n".join(lines)


def parse_pdf(content_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")

    return "\n\n".join(pages) if pages else "Could not extract text from PDF."


def parse_file(filename: str, content_bytes: bytes) -> tuple[str, str]:
    """Parse a file and return (text_content, file_type)."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".docx":
        return parse_docx(content_bytes), "docx"
    elif ext == ".xlsx":
        return parse_xlsx(content_bytes), "xlsx"
    elif ext == ".pdf":
        return parse_pdf(content_bytes), "pdf"
    elif ext in (".txt", ".md", ".csv", ".json", ".py", ".yaml", ".yml"):
        return content_bytes.decode("utf-8", errors="replace"), ext.lstrip(".")
    else:
        return f"Unsupported file type: {ext}", "unknown"


# ── File Editing ──────────────────────────────────────────────────────


def create_edited_docx(original_bytes: bytes, instructions: str, new_content: str) -> bytes:
    """Create an edited .docx file by replacing content."""
    from docx import Document

    doc = Document(io.BytesIO(original_bytes))

    # Clear existing paragraphs and replace with new content
    # Preserve formatting from the first paragraph style
    first_style = doc.paragraphs[0].style if doc.paragraphs else None

    for para in doc.paragraphs:
        para.clear()

    # Add new content as paragraphs
    for i, line in enumerate(new_content.split("\n")):
        if i < len(doc.paragraphs):
            doc.paragraphs[i].text = line
            if first_style:
                doc.paragraphs[i].style = first_style
        else:
            doc.add_paragraph(line)

    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()


def create_edited_xlsx(original_bytes: bytes, sheet_name: str, updates: list[dict]) -> bytes:
    """Apply updates to an .xlsx file.

    updates: list of {row, col, value} dicts (1-indexed)
    """
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(original_bytes))
    ws = wb[sheet_name] if sheet_name in wb.sheetnames else wb.active

    for update in updates:
        row = update.get("row", 1)
        col = update.get("col", 1)
        value = update.get("value", "")
        ws.cell(row=row, column=col, value=value)

    output = io.BytesIO()
    wb.save(output)
    return output.getvalue()


# ── Agent Tools ───────────────────────────────────────────────────────


@tool
def get_uploaded_file_content(conversation_id: str) -> str:
    """Get the text content of a file uploaded by the user in this conversation.

    Args:
        conversation_id: The current conversation ID (from context brackets).
    """
    file_info = _uploaded_files.get(conversation_id)
    if not file_info:
        return "No file has been uploaded in this conversation."

    filename = file_info["filename"]
    text = file_info["text_content"]
    file_type = file_info["file_type"]

    if len(text) > 30000:
        text = text[:30000] + f"\n\n... [truncated at 30,000 chars, full file is {len(text)} chars]"

    return f"**File: {filename}** (type: {file_type})\n\n{text}"


@tool
def edit_document(
    conversation_id: str,
    user_email: str,
    new_content: str,
    summary: str = "Document edited by SamurAI",
) -> str:
    """Edit the uploaded document and send the modified version back.

    Only works with .docx files. Creates a new version with the provided content
    and sends it back to the user via Teams file upload.

    Args:
        conversation_id: The current conversation ID (from context brackets).
        user_email: The user's email (from context brackets).
        new_content: The full new content for the document.
        summary: Brief description of the changes made.
    """
    file_info = _uploaded_files.get(conversation_id)
    if not file_info:
        return "No file has been uploaded in this conversation to edit."

    file_type = file_info["file_type"]
    if file_type not in ("docx", "xlsx"):
        return f"Cannot edit {file_type} files. Only .docx and .xlsx files can be edited."

    try:
        if file_type == "docx":
            edited_bytes = create_edited_docx(
                file_info["content_bytes"], summary, new_content
            )
        else:
            return "For .xlsx edits, use edit_spreadsheet instead."

        filename = file_info["filename"]
        edited_name = f"edited_{filename}"

        # Store for FileConsentCard upload
        _pending_edited_files[conversation_id] = {
            "filename": edited_name,
            "content_bytes": edited_bytes,
            "summary": summary,
        }

        return f"Document edited. The modified file '{edited_name}' will be sent to you for review."
    except Exception as e:
        return f"Error editing document: {e}"


@tool
def edit_spreadsheet(
    conversation_id: str,
    user_email: str,
    sheet_name: str,
    updates: str,
    summary: str = "Spreadsheet edited by SamurAI",
) -> str:
    """Edit the uploaded spreadsheet and send the modified version back.

    Only works with .xlsx files.

    Args:
        conversation_id: The current conversation ID (from context brackets).
        user_email: The user's email (from context brackets).
        sheet_name: The sheet to edit (use the sheet name from get_uploaded_file_content).
        updates: JSON string of updates as a list of objects with row, col, value.
                 Example: '[{"row": 2, "col": 3, "value": "new value"}]'
                 Rows and columns are 1-indexed.
        summary: Brief description of the changes made.
    """
    import json as _json

    file_info = _uploaded_files.get(conversation_id)
    if not file_info:
        return "No file has been uploaded in this conversation to edit."

    if file_info["file_type"] != "xlsx":
        return "This tool only works with .xlsx files."

    try:
        update_list = _json.loads(updates)
        edited_bytes = create_edited_xlsx(
            file_info["content_bytes"], sheet_name, update_list
        )

        filename = file_info["filename"]
        edited_name = f"edited_{filename}"

        _pending_edited_files[conversation_id] = {
            "filename": edited_name,
            "content_bytes": edited_bytes,
            "summary": summary,
        }

        return f"Spreadsheet edited ({len(update_list)} cells updated). The modified file '{edited_name}' will be sent to you."
    except Exception as e:
        return f"Error editing spreadsheet: {e}"


FILE_HANDLER_TOOLS = [
    get_uploaded_file_content,
    edit_document,
    edit_spreadsheet,
]
