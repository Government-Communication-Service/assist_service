"""
Summarise a pytest run from its JUnit XML (and optionally a coverage XML report) as Markdown.

Prints the summary, and appends it to the GitHub Actions job summary when $GITHUB_STEP_SUMMARY is set.

Usage: python scripts/ci/test_summary.py test-output/test-output.xml [test-output/coverage.xml]
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SLOWEST = 10
MAX_FAILURES = 30
MESSAGE_CHARS = 300


def _test_id(case: ET.Element) -> str:
    """The pytest node id, e.g. tests/chat/test_x.py::TestClass::test_name, so it can be pasted into pytest."""
    parts = case.get("classname", "").split(".")
    # JUnit joins the module path and any test classes with dots; classes are the CamelCase parts.
    module = [part for part in parts if part and not part[0].isupper()]
    classes = [part for part in parts if part and part[0].isupper()]
    if not module:
        return case.get("name", "")
    return "::".join(["/".join(module) + ".py", *classes, case.get("name", "")])


def _one_line(text: str | None) -> str:
    line = " ".join((text or "").split())
    return line if len(line) <= MESSAGE_CHARS else line[: MESSAGE_CHARS - 1] + "…"


def summarise_tests(junit_path: Path) -> list[str]:
    cases = list(ET.parse(junit_path).getroot().iter("testcase"))
    failed, skipped = [], []
    for case in cases:
        for tag in ("failure", "error"):
            if (problem := case.find(tag)) is not None:
                failed.append((tag, _test_id(case), _one_line(problem.get("message") or problem.text)))
        if (skip := case.find("skipped")) is not None:
            skipped.append((_test_id(case), _one_line(skip.get("message"))))
    passed = len(cases) - len({test_id for _, test_id, _ in failed}) - len(skipped)
    total_seconds = sum(float(case.get("time") or 0) for case in cases)

    icon = "❌" if failed else "✅"
    lines = [
        f"## {icon} Tests: {passed} passed, {len(failed)} failed, {len(skipped)} skipped",
        "",
        f"Total test time: {total_seconds:.0f}s across {len(cases)} tests (summed over parallel workers).",
        "",
    ]
    if failed:
        lines += ["### Failures", "", "| Test | Error |", "| --- | --- |"]
        for tag, test_id, message in failed[:MAX_FAILURES]:
            label = " (error in setup/teardown)" if tag == "error" else ""
            message = message.replace("|", "\\|")
            lines.append(f"| `{test_id}`{label} | {message} |")
        if len(failed) > MAX_FAILURES:
            lines.append(f"| … and {len(failed) - MAX_FAILURES} more | See the job log |")
        lines.append("")
    if skipped:
        lines += ["<details><summary>Skipped tests</summary>", ""]
        lines += [f"- `{test_id}`: {reason}" for test_id, reason in skipped]
        lines += ["", "</details>", ""]

    slowest = sorted(cases, key=lambda case: float(case.get("time") or 0), reverse=True)[:SLOWEST]
    lines += ["<details><summary>Slowest tests</summary>", "", "| Time | Test |", "| ---: | --- |"]
    lines += [f"| {float(case.get('time') or 0):.1f}s | `{_test_id(case)}` |" for case in slowest]
    lines += ["", "</details>", ""]
    return lines


def summarise_coverage(coverage_path: Path) -> list[str]:
    root = ET.parse(coverage_path).getroot()
    # File names are relative to the measured source directory; show them relative to the repository.
    source = Path(root.findtext("sources/source") or ".")
    rate = float(root.get("line-rate", 0)) * 100
    lines = [f"**Line coverage:** {rate:.1f}% ({root.get('lines-covered')} of {root.get('lines-valid')} lines)", ""]

    files = []
    for cls in root.iter("class"):
        statements = len(cls.findall("lines/line"))
        if statements >= 20:
            name = os.path.relpath(source / cls.get("filename"))
            files.append((float(cls.get("line-rate", 0)) * 100, statements, name))
    least_covered = sorted(files)[:10]
    if least_covered:
        lines += ["<details><summary>Least covered files (20+ lines)</summary>", ""]
        lines += ["| Coverage | Lines | File |", "| ---: | ---: | --- |"]
        lines += [f"| {rate:.0f}% | {statements} | `{name}` |" for rate, statements, name in least_covered]
        lines += ["", "</details>", ""]
    return lines


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    junit_path = Path(argv[1])
    if not junit_path.exists():
        print(f"No test report at {junit_path}; the tests didn't run.", file=sys.stderr)
        return 1

    lines = summarise_tests(junit_path)
    if len(argv) == 3 and Path(argv[2]).exists():
        lines += summarise_coverage(Path(argv[2]))
    markdown = "\n".join(lines)

    print(markdown)
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
