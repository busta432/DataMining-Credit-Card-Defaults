"""Word/table size of the capped body (Introduction -> References) for two builds.

The page cap excludes everything from the "Appendix" heading onward. Every paragraph
carries style 'Normal', so the Appendix boundary is found by text, not by style.
"""
import re
import sys
from docx import Document

STOP = re.compile(r"^Appendix\s*$")


def measure(path):
    doc = Document(path)
    words = 0
    for p in doc.paragraphs:
        t = p.text.strip()
        if STOP.match(t):
            break
        words += len(t.split())
    rows = sum(len(t.rows) for t in doc.tables)
    return words, rows, len(doc.tables)


for p in sys.argv[1:]:
    w, r, n = measure(p)
    print(f"{p:28} body_words={w:6d}  all_table_rows={r:4d}  tables={n:3d}")
