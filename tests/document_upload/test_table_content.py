from types import SimpleNamespace

from unstructured.documents.elements import ElementMetadata, Table

from app.document_upload.table_content import (
    build_table_chunk_documents,
    html_table_to_markdown,
    split_markdown_table,
)


def test_html_table_to_markdown_converts_basic_table():
    html = "<table><tr><td>Name</td><td>Dept</td></tr><tr><td>Alice</td><td>Eng</td></tr></table>"

    markdown = html_table_to_markdown(html)

    lines = markdown.split("\n")
    assert lines[0] == "| Name | Dept |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| Alice | Eng |"


def test_html_table_to_markdown_handles_colspan():
    html = '<table><tr><td colspan="2">Header</td></tr><tr><td>a</td><td>b</td></tr></table>'

    markdown = html_table_to_markdown(html)

    assert markdown.split("\n")[0] == "| Header | Header |"


def test_html_table_to_markdown_fallback_on_malformed_html():
    html = "<table><tr><td>unterminated"

    markdown = html_table_to_markdown(html)

    assert "unterminated" in markdown


def test_html_table_to_markdown_fallback_when_no_table_tag():
    html = "<p>just some text, no table</p>"

    markdown = html_table_to_markdown(html)

    assert markdown == "just some text, no table"


def test_html_table_to_markdown_returns_fallback_when_table_has_no_rows():
    html = "<table></table>"

    markdown = html_table_to_markdown(html)

    assert markdown == ""


def test_html_table_to_markdown_returns_fallback_when_all_rows_have_no_cells():
    html = "<table><tr></tr><tr></tr></table>"

    markdown = html_table_to_markdown(html)

    assert markdown == ""


def test_html_table_to_markdown_falls_back_on_invalid_colspan_attribute():
    """
    A malformed colspan (non-numeric) makes int() genuinely raise inside the parsing
    loop - exercises the outer except Exception fallback, not just BeautifulSoup's own
    (very lenient) recovery from unterminated tags.
    """
    html = '<table><tr><td colspan="abc">x</td></tr></table>'

    markdown = html_table_to_markdown(html)

    assert markdown == "x"


def test_split_markdown_table_returns_unchanged_when_under_limit():
    markdown = "| a | b |\n| --- | --- |\n| 1 | 2 |"

    pieces = split_markdown_table(markdown, max_chars=1000)

    assert pieces == [markdown]


def test_split_markdown_table_repeats_header_and_respects_max_chars():
    header = "| col1 | col2 |"
    separator = "| --- | --- |"
    rows = [f"| row{i} | val{i} |" for i in range(50)]
    markdown = "\n".join([header, separator] + rows)

    pieces = split_markdown_table(markdown, max_chars=200)

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith(f"{header}\n{separator}\n")
        assert len(piece) <= 200 or piece.count("\n") == 2  # a single oversized row is kept whole


def test_split_markdown_table_never_truncates_a_single_oversized_row():
    header = "| col |"
    separator = "| --- |"
    huge_row = "| " + ("x" * 500) + " |"
    markdown = "\n".join([header, separator, huge_row])

    pieces = split_markdown_table(markdown, max_chars=50)

    assert len(pieces) == 1
    assert huge_row in pieces[0]


def test_split_markdown_table_returns_unchanged_when_no_separator_row():
    markdown = "not a table\n" + ("x" * 2000)

    pieces = split_markdown_table(markdown, max_chars=100)

    assert pieces == [markdown]


def test_html_table_to_markdown_without_header_emits_no_separator_row():
    """
    Regression test: continuation chunks (is_continuation=True) start mid-table at an
    arbitrary data row. Rendering with has_header=False must not promote that row to a
    header or emit a '| --- |' separator, otherwise an LLM reading the chunk in isolation
    treats a data value (e.g. a row's ID) as a column name.
    """
    html = "<table><tr><td>row20</td><td>val20</td></tr><tr><td>row21</td><td>val21</td></tr></table>"

    markdown = html_table_to_markdown(html, has_header=False)

    lines = markdown.split("\n")
    assert lines == ["| row20 | val20 |", "| row21 | val21 |"]
    assert "---" not in markdown


def test_split_markdown_table_without_header_splits_without_repeating_anything():
    rows = [f"| row{i} | val{i} |" for i in range(50)]
    markdown = "\n".join(rows)

    pieces = split_markdown_table(markdown, max_chars=200, has_header=False)

    assert len(pieces) > 1
    assert "".join(pieces).replace("\n", "|").count("row0") == 1  # no row duplicated across pieces
    for piece in pieces:
        assert "---" not in piece


def _make_table_element(html: str, is_continuation: bool = False) -> Table:
    return Table(text="ignored", metadata=ElementMetadata(text_as_html=html, is_continuation=is_continuation))


def test_build_table_chunk_documents_single_piece():
    element = _make_table_element("<table><tr><td>a</td><td>b</td></tr></table>")
    document = SimpleNamespace(name="doc.xlsx", url="http://example.com/doc.xlsx", uuid="doc-uuid")

    documents = build_table_chunk_documents(element, document, sanitize=lambda s: s, max_chars=1000)

    assert len(documents) == 1
    assert documents[0]["chunk_name"] == "Table"
    assert "| a | b |" in documents[0]["chunk_content"]


def test_build_table_chunk_documents_splits_and_numbers_parts():
    header = "<tr><td>col1</td><td>col2</td></tr>"
    rows = "".join(f"<tr><td>row{i}</td><td>val{i}</td></tr>" for i in range(50))
    element = _make_table_element(f"<table>{header}{rows}</table>")
    document = SimpleNamespace(name="doc.xlsx", url="http://example.com/doc.xlsx", uuid="doc-uuid")

    documents = build_table_chunk_documents(element, document, sanitize=lambda s: s, max_chars=200)

    assert len(documents) > 1
    assert documents[0]["chunk_name"] == "Table (part 1/%d)" % len(documents)
    for doc in documents:
        assert "col1" in doc["chunk_content"]  # header repeated in every part


def test_build_table_chunk_documents_drops_pieces_that_sanitize_to_empty():
    element = _make_table_element("<table><tr><td>a</td></tr></table>")
    document = SimpleNamespace(name="doc.xlsx", url="http://example.com/doc.xlsx", uuid="doc-uuid")

    documents = build_table_chunk_documents(element, document, sanitize=lambda s: "", max_chars=1000)

    assert documents == []


def test_build_table_chunk_documents_numbers_parts_without_gaps_when_middle_piece_dropped():
    """
    Regression test: part numbering must be computed after dropping empty-sanitized
    pieces, not before - otherwise dropping the middle piece of three produces
    "part 1/3" and "part 3/3" with no "2/3".
    """
    header = "<tr><td>col1</td><td>col2</td></tr>"
    rows = "".join(f"<tr><td>row{i}</td><td>val{i}</td></tr>" for i in range(50))
    element = _make_table_element(f"<table>{header}{rows}</table>")
    document = SimpleNamespace(name="doc.xlsx", url="http://example.com/doc.xlsx", uuid="doc-uuid")

    call_count = 0

    def sanitize_dropping_second_call(s: str) -> str:
        nonlocal call_count
        call_count += 1
        return "" if call_count == 2 else s

    documents = build_table_chunk_documents(element, document, sanitize=sanitize_dropping_second_call, max_chars=200)

    part_numbers = [doc["chunk_name"].split("part ")[1].rstrip(")") for doc in documents]
    assert part_numbers == [f"{i + 1}/{len(documents)}" for i in range(len(documents))]


def test_build_table_chunk_documents_continuation_chunk_has_no_fabricated_header():
    """
    Regression test: a continuation chunk (is_continuation=True, as set by unstructured's
    by_title chunking on tables split at an arbitrary data row) must not have its first
    data row promoted to a fabricated Markdown header.
    """
    html = "<table><tr><td>row20</td><td>val20</td></tr><tr><td>row21</td><td>val21</td></tr></table>"
    element = _make_table_element(html, is_continuation=True)
    document = SimpleNamespace(name="doc.pdf", url="http://example.com/doc.pdf", uuid="doc-uuid")

    documents = build_table_chunk_documents(element, document, sanitize=lambda s: s, max_chars=1000)

    assert len(documents) == 1
    assert "---" not in documents[0]["chunk_content"]
    assert documents[0]["chunk_content"] == "| row20 | val20 |\n| row21 | val21 |"
