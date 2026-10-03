"""One-shot: move SHAP after the hold-out, lift fairness out of the capped body."""
from pathlib import Path

p = Path("phase2_report.md")
text = p.read_text(encoding="utf-8")


def cut(start_marker, end_marker):
    """Remove and return the span [start_marker, end_marker), exclusive of end."""
    global text
    i = text.index(start_marker)
    j = text.index(end_marker, i + len(start_marker))
    block = text[i:j]
    text = text[:i] + text[j:]
    return block


shap = cut("### 5.5 SHAP — answering Q2", "### 5.6 Fairness")
fairness = cut("### 5.6 Fairness", "### 5.7 Final hold-out evaluation")

# The hold-out becomes 5.3; SHAP follows it as 5.4.
text = text.replace(
    "### 5.7 Final hold-out evaluation",
    "### 5.3 Confusion matrix and Total Expected Cost on unseen data",
    1,
)

shap = shap.replace("### 5.5 SHAP — answering Q2", "### 5.4 SHAP — answering Q2", 1)

anchor = "climb comes from being an ensemble than from being a well-tuned one** (§4.1, C.5).\n"
i = text.index(anchor) + len(anchor)
text = text[:i] + "\n" + shap.rstrip() + "\n" + text[i:]

Path("_fairness_block.md").write_text(fairness, encoding="utf-8")
p.write_text(text, encoding="utf-8")
print("moved SHAP ->5.4, extracted fairness block", len(fairness), "chars")
