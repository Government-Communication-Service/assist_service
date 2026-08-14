import logging
from typing import Callable

from bs4 import BeautifulSoup
from unstructured.documents.elements import Table

from app.opensearch.schemas import OpenSearchRecord

logger = logging.getLogger()


def html_table_to_markdown(html: str, has_header: bool = True) -> str:
    """
    Converts an HTML table (as produced by unstructured's `text_as_html`) into a
    Markdown pipe-table. Colspan is handled by repeating the cell's text across the
    spanned columns. When has_header is True the first row is treated as the header
    row and a separator row is emitted beneath it - pass has_header=False for
    continuation chunks (element.metadata.is_continuation), which start mid-table at
    an arbitrary data row and would otherwise have that row wrongly promoted to a
    header. Falls back to plain text extraction if the HTML has no parseable table or
    parsing otherwise fails, so a markup quirk never raises out of the ingestion path.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            return soup.get_text(separator=" ", strip=True)

        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells: list[str] = []
            for cell in tr.find_all(["td", "th"]):
                text = cell.get_text(separator=" ", strip=True)
                colspan = int(cell.get("colspan", 1) or 1)
                cells.extend([text] * max(colspan, 1))
            if cells:
                rows.append(cells)

        if not rows:
            return soup.get_text(separator=" ", strip=True)

        num_cols = max(len(row) for row in rows)
        for row in rows:
            row.extend([""] * (num_cols - len(row)))

        def _format_row(row: list[str]) -> str:
            cells = (cell.replace("|", "\\|").replace("\n", " ") for cell in row)
            return "| " + " | ".join(cells) + " |"

        if not has_header:
            return "\n".join(_format_row(row) for row in rows)

        header, *data_rows = rows
        lines = [_format_row(header), "| " + " | ".join(["---"] * num_cols) + " |"]
        lines.extend(_format_row(row) for row in data_rows)
        return "\n".join(lines)
    except Exception:
        logger.warning("Failed to convert HTML table to markdown, falling back to plain text", exc_info=True)
        return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def split_markdown_table(markdown: str, max_chars: int, has_header: bool = True) -> list[str]:
    """
    Splits a Markdown pipe-table's data rows into groups so each returned chunk stays
    under max_chars. When has_header is True, the header + separator row are repeated
    in each chunk (pass has_header=False for a continuation table with no header row
    to repeat - see html_table_to_markdown). A single row that alone exceeds max_chars
    is kept whole rather than truncated. Returns [markdown] unchanged if it's already
    small enough or (when has_header) has no detectable header/separator rows.
    """
    if len(markdown) <= max_chars:
        return [markdown]

    lines = markdown.split("\n")
    if has_header:
        if len(lines) < 2 or not lines[1].strip("| -").strip() == "":
            return [markdown]
        header, separator, *data_rows = lines
        prefix = f"{header}\n{separator}\n"
    else:
        data_rows = lines
        prefix = ""

    pieces: list[str] = []
    current_rows: list[str] = []
    current_len = len(prefix)
    for row in data_rows:
        row_len = len(row) + 1  # + newline
        if current_rows and current_len + row_len > max_chars:
            pieces.append(prefix + "\n".join(current_rows))
            current_rows = []
            current_len = len(prefix)
        current_rows.append(row)
        current_len += row_len

    if current_rows:
        pieces.append(prefix + "\n".join(current_rows))

    return pieces or [markdown]


def build_table_chunk_documents(
    element: Table,
    document,
    sanitize: Callable[[str], str],
    max_chars: int,
) -> list[dict]:
    """
    Converts one Table element's text_as_html into one or more OpenSearch-ready
    dicts, splitting into multiple parts if the resulting markdown table exceeds
    max_chars. Returns [] if all resulting content sanitizes to empty.

    Continuation chunks (element.metadata.is_continuation, set by unstructured's
    by_title chunking on tables split at an arbitrary data row) are rendered without
    a fabricated header row.
    """
    has_header = not element.metadata.is_continuation
    markdown = html_table_to_markdown(element.metadata.text_as_html, has_header=has_header)
    pieces = split_markdown_table(markdown, max_chars, has_header=has_header)

    contents = [content for content in (sanitize(piece) for piece in pieces) if content]
    name_base = sanitize(element.category)
    result = []
    for i, content in enumerate(contents):
        name = f"{name_base} (part {i + 1}/{len(contents)})" if len(contents) > 1 else name_base
        record = OpenSearchRecord(document.name, document.url, name, content, document.uuid)
        result.append(record.to_opensearch_dict())
    return result
