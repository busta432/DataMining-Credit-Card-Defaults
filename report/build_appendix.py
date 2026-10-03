"""Expand the Appendix B placeholder in assignment2_submission.md with every source file.

    python report/build_appendix.py

Rewrites the marker line in place, so re-running is idempotent.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TARGET = ROOT / "assignment2_submission.md"
MARKER = "<!-- INSERT FULL SOURCE LISTING HERE -->"

ORDER = [
    ("src", ["config.py", "data.py", "models.py", "runner.py", "tuning.py",
             "metrics.py", "costs.py", "plotting.py", "report.py"]),
    ("experiments", [f"exp{n:02d}_{s}.py" for n, s in [
        (1, "ladder"), (2, "tuning"), (3, "imbalance"), (4, "encoding"),
        (5, "engineered"), (6, "groups"), (7, "recency"), (8, "learning_curve"),
        (9, "calibration"), (10, "cost"), (11, "shap"), (12, "stability"),
        (13, "fairness"), (14, "final")]] + ["aggregate.py", "figures.py", "verify.py"]),
]


def listing() -> str:
    out = []
    for pkg, names in ORDER:
        out.append(f"\n### {pkg}/\n")
        for name in names:
            path = REPO / pkg / name
            if not path.exists():
                out.append(f"\n**`{pkg}/{name}`** — not found, skipped.\n")
                continue
            out.append(f"\n**`{pkg}/{name}`**\n\n```python\n"
                       f"{path.read_text(encoding='utf-8').rstrip()}\n```\n")
    return "".join(out)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    head, sep, _ = text.partition(MARKER)
    if not sep:
        raise SystemExit(f"marker not found in {TARGET.name}; already expanded?")
    TARGET.write_text(head + MARKER + "\n" + listing(), encoding="utf-8")
    print(f"expanded -> {TARGET}")


if __name__ == "__main__":
    main()
