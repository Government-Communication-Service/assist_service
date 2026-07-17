#!/usr/bin/env python
"""
generate_dag.py — visualise the Alembic migration DAG

Generates a PNG of the migration history graph using Graphviz.

Usage:
    python scripts/migrations/generate_dag.py [output.png]

    Output path defaults to a temporary file (auto-cleaned by the OS).

Requirements:
    - graphviz must be installed: brew install graphviz
    - Run from the repo root, or any directory; the script resolves paths automatically.

The graph shows each migration as a node, with edges pointing from parent to
child. Merge migrations (with multiple parents) will appear as nodes with
multiple incoming edges, making branch/merge points easy to spot.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_DIR = REPO_ROOT / "app" / "alembic"


def build_dot() -> str:
    sys.path.insert(0, str(REPO_ROOT))

    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(ALEMBIC_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(ALEMBIC_DIR))
    script = ScriptDirectory.from_config(cfg)

    edges = []
    for rev in script.walk_revisions():
        downs = rev.down_revision
        if downs is None:
            continue
        if isinstance(downs, str):
            downs = (downs,)
        for d in downs:
            label = (rev.doc or rev.revision)[:35]
            edges.append(f'  "{d[:8]}" -> "{rev.revision[:8]}" [label="{label}"]')

    return "digraph migrations {\n" + "\n".join(edges) + "\n}\n"


def main() -> None:
    if len(sys.argv) > 1:
        output = Path(sys.argv[1])
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        output = Path(tmp.name)

    dot_src = build_dot()

    result = subprocess.run(
        ["dot", "-Tpng", "-o", str(output)],
        input=dot_src,
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print("graphviz error:", result.stderr, file=sys.stderr)
        print("Is graphviz installed? Try: brew install graphviz", file=sys.stderr)
        sys.exit(1)

    print(f"Written to {output.resolve()}")

    if sys.platform == "darwin":
        subprocess.run(["open", str(output)])


if __name__ == "__main__":
    main()
