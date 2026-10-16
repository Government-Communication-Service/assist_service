# /// script
# requires-python = ">=3.12"
# dependencies = ["python-docx>=1.1"]
# ///
# ruff: noqa: E501, E731, E741, C408, B008, B023
"""Write the Word document listing every evaluation case, its questions and expected answers.

Run from scripts/evaluation:
    uv run build_cases_doc.py
The output replaces data/ignored/evaluation/evaluation-cases.docx. If that file exists it is used as the template
(styles, page setup and footer) and is first copied to evaluation-cases-<today>.docx; otherwise a plain A4 document is made.
"""

import argparse
import json
import re
import shutil
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import docx
from docx.oxml import parse_xml
from docx.shared import Inches

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--cases-dir", type=Path, default=HERE / "cases")
parser.add_argument("--output", type=Path, default=HERE.parents[1] / "data/ignored/evaluation/evaluation-cases.docx")
args = parser.parse_args()
CASES, DOC = f"{args.cases_dir}/", args.output

SCENARIOS = [
    ("central_rag", "Central guidance (RAG)", "Questions answered from the central GCS guidance index."),
    ("document_docx", "Uploaded Word documents", "Questions about a .docx the user uploads."),
    ("document_odt", "Uploaded OpenDocument text files", "Questions about an .odt the user uploads."),
    (
        "document_pptx",
        "Uploaded PowerPoint decks",
        "Questions about decks, including ones that depend on where things sit on a slide.",
    ),
    (
        "document_summary",
        "Document summaries",
        "Requests to summarise an uploaded document within a constraint such as a word limit.",
    ),
    ("document_xlsx", "Uploaded Excel workbooks", "Questions about figures in spreadsheets."),
    ("govuk", "GOV.UK search", "Questions that need current GOV.UK content."),
    (
        "long_chat",
        "Long conversation",
        "One long chat that mixes document, guidance, GOV.UK and Smart Targets questions.",
    ),
    ("no_tool", "No retrieval needed", "Questions the model can answer without searching or reading a document."),
    ("smart_targets", "Smart Targets", "Questions that use the GCS Data API through Smart Targets."),
]
SCEN = [s[0] for s in SCENARIOS]
HEAD = {s[0]: (s[1], s[2]) for s in SCENARIOS}

if DOC.exists():
    backup = DOC.with_name(f"{DOC.stem}-{date.today().isoformat()}{DOC.suffix}")
    if not backup.exists():
        shutil.copy(DOC, backup)
    d = docx.Document(str(DOC))
else:
    d = docx.Document()
    section = d.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.left_margin = section.right_margin = Inches(0.79)

NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
body = d.element.body
sect = body[-1]
for ch in list(body)[:-1]:
    body.remove(ch)


def add(xml):
    sect.addprevious(parse_xml(xml.replace("<w:p>", f"<w:p {NS}>", 1).replace("<w:tbl>", f"<w:tbl {NS}>", 1)))


def run(text, rpr=""):
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def para(text, rpr, ppr='<w:spacing w:after="60"/>', style=None):
    st = f'<w:pStyle w:val="{style}"/>' if style else ""
    return f"<w:p><w:pPr>{st}{ppr}</w:pPr>{run(text, rpr)}</w:p>"


SZ = lambda n: f'<w:sz w:val="{n}"/><w:szCs w:val="{n}"/>'
BORD = "".join(f'<w:{s} w:val="single" w:color="BFBFBF" w:sz="4"/>' for s in ("top", "left", "bottom", "right"))
MAR = "".join(
    f'<w:{s} w:type="dxa" w:w="{w}"/>' for s, w in (("top", 70), ("left", 110), ("bottom", 70), ("right", 110))
)


def cell(w, paras, fill=None):
    shd = f'<w:shd w:fill="{fill}" w:color="auto" w:val="clear"/>' if fill else ""
    return f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/><w:tcBorders>{BORD}</w:tcBorders>{shd}<w:tcMar>{MAR}</w:tcMar></w:tcPr>{"".join(paras)}</w:tc>'


cp = lambda text, rpr=SZ(20): f'<w:p><w:pPr><w:spacing w:after="60"/></w:pPr>{run(text, rpr)}</w:p>'

FLAG = {"use_rag": "central guidance", "use_gov_uk_search_api": "GOV.UK search", "use_smart_targets": "Smart Targets"}
data = {}
for s in SCEN:
    data[s] = json.load(open(CASES + s + ".json"))["cases"]
n_cases = sum(len(v) for v in data.values())
n_q = sum(len(c["turns"]) for v in data.values() for c in v)
n_skip = sum(1 for v in data.values() for c in v if c.get("skip"))

add(f'<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr>{run("Evaluation cases and expected answers")}</w:p>')
add(
    para(
        f"{n_cases} cases, {n_q} questions across {len(SCEN)} scenarios. Each case is one conversation: the questions are asked in order in the same chat, and each answer is compared with the expected answer shown here.",
        SZ(22),
    )
)
add(
    para(
        'Cases marked "not yet verified" still need their figures checking against the source data. Cases marked "currently excluded" are not run at the moment.',
        SZ(22),
    )
)
for s in SCEN:
    title, desc = HEAD[s]
    add(f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>{run(title)}</w:p>')
    add(para(desc, '<w:i/><w:iCs/><w:color w:val="555555"/>' + SZ(20), '<w:keepNext/><w:spacing w:after="60"/>'))
    for c in data[s]:
        add(f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr>{run(c["id"])}</w:p>')
        tools = ", ".join(FLAG[f] for f in c["needed_flags"]) or "none needed"
        line = f"Tools switched on: {tools}" + (f"   |   Currently excluded: {c['skip']}" if c.get("skip") else "")
        add(para(line, '<w:color w:val="555555"/>' + SZ(18), '<w:keepNext/><w:spacing w:after="100"/>'))
        hdr = lambda w, t: cell(w, [cp(t, "<w:b/><w:bCs/>" + SZ(20))], "E8EEF4")
        rows = [
            f"<w:tr><w:trPr><w:tblHeader/></w:trPr>{hdr(700, '#')}{hdr(3900, 'Question')}{hdr(5038, 'Expected answer')}</w:tr>"
        ]
        for i, t in enumerate(c["turns"], 1):
            qp = [cp(t["query"])]
            if t.get("add_documents"):
                qp.append(
                    cp(
                        "Uploads: " + ", ".join(t["add_documents"]),
                        "<w:i/><w:iCs/>" + '<w:color w:val="555555"/>' + SZ(18),
                    )
                )
            ans = t["expected_answer"]
            todo = "TODO" in ans or not ans.strip()
            ans = re.sub(r"^TODO\(verify[^)]*\):\s*", "", ans).strip()
            ap = [cp(ans or "(no expected answer yet)")]
            if todo:
                ap.append(
                    cp("Not yet verified against the source data.", '<w:b/><w:bCs/><w:color w:val="B45309"/>' + SZ(18))
                )
            rows.append(
                f"<w:tr>{cell(700, [cp(str(i))])}{cell(3900, qp)}{cell(5038, ap, 'FEF6E7' if todo else None)}</w:tr>"
            )
        grid = '<w:tblGrid><w:gridCol w:w="700"/><w:gridCol w:w="3900"/><w:gridCol w:w="5038"/></w:tblGrid>'
        borders = "".join(
            f'<w:{s} w:val="single" w:color="auto" w:sz="4"/>'
            for s in ("top", "left", "bottom", "right", "insideH", "insideV")
        )
        add(
            f'<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="9638"/><w:tblBorders>{borders}</w:tblBorders></w:tblPr>{grid}{"".join(rows)}</w:tbl>'
        )
DOC.parent.mkdir(parents=True, exist_ok=True)
d.save(str(DOC))
print(n_cases, n_q, n_skip)
