"""Load and validate evaluation case files (see thoughts/agent_loop/evaluation-case-format.md)."""

import json
from dataclasses import dataclass
from pathlib import Path

FLAGS = ("use_rag", "use_gov_uk_search_api", "use_smart_targets")


@dataclass(frozen=True)
class Turn:
    query: str
    expected_answer: str
    add_documents: tuple[str, ...] = ()


@dataclass(frozen=True)
class Case:
    id: str
    scenario: str
    needed_flags: tuple[str, ...]
    turns: tuple[Turn, ...]


def load_cases(paths: list[Path], documents_dir: Path) -> list[Case]:
    """Load cases from JSON files, validating flags, documents and ids.

    Args:
        paths: Case files, in the order their cases should run.
        documents_dir: Directory holding the files named in `add_documents`.

    Returns:
        The cases in file order. A case with a non-empty "skip" field (the reason) is left out.

    Raises:
        ValueError: If a case is malformed or an id is duplicated.
    """
    cases: list[Case] = []
    seen: set[str] = set()
    for path in paths:
        data = json.loads(path.read_text())
        scenario = data["scenario"]
        for raw in data["cases"]:
            if raw.get("skip"):
                print(f"Skipping {raw['id']}: {raw['skip']}")
                continue
            case = _parse_case(raw, scenario, documents_dir, path)
            if case.id in seen:
                raise ValueError(f"{path}: duplicate case id {case.id}")
            seen.add(case.id)
            cases.append(case)
    return cases


def _parse_case(raw: dict, scenario: str, documents_dir: Path, path: Path) -> Case:
    case_id = raw["id"]
    needed_flags = tuple(raw.get("needed_flags", []))
    unknown_flags = set(needed_flags) - set(FLAGS)
    if unknown_flags:
        raise ValueError(f"{path}: {case_id} has unknown flags {sorted(unknown_flags)}")
    if not raw.get("turns"):
        raise ValueError(f"{path}: {case_id} has no turns")

    turns = []
    for index, raw_turn in enumerate(raw["turns"]):
        query = raw_turn.get("query", "").strip()
        expected_answer = raw_turn.get("expected_answer", "").strip()
        if not query or not expected_answer:
            raise ValueError(f"{path}: {case_id} turn {index} needs a query and an expected_answer")
        add_documents = tuple(raw_turn.get("add_documents", []))
        for name in add_documents:
            if not (documents_dir / name).is_file():
                raise ValueError(f"{path}: {case_id} turn {index} references missing document {name}")
        turns.append(Turn(query=query, expected_answer=expected_answer, add_documents=add_documents))
    return Case(id=case_id, scenario=scenario, needed_flags=needed_flags, turns=tuple(turns))
