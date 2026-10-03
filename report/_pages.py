"""Report total pages and where the capped body ends, via an isolated Word instance.

DispatchEx forces a private Word process so an already-open document in the user's
session is never touched. The document is opened read-only and closed without saving.
"""
import sys
from pathlib import Path
import win32com.client as win32

WD_PAGES = 2            # wdStatisticPages
WD_END_PAGE = 3         # wdActiveEndPageNumber

path = str(Path(sys.argv[1]).resolve())
word = win32.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = False
try:
    doc = word.Documents.Open(path, ReadOnly=True, AddToRecentFiles=False)
    total = doc.ComputeStatistics(WD_PAGES)   # paginates as a side effect

    appendix_page = None
    for p in doc.Paragraphs:
        t = p.Range.Text.strip()
        if t == "Appendix":
            appendix_page = p.Range.Information(WD_END_PAGE)
            break

    print(f"total pages:       {total}")
    if appendix_page:
        print(f"appendix starts:   p{appendix_page}")
        print(f"capped body:       {appendix_page - 1} pages (Introduction -> References)")
    doc.Close(False)
finally:
    word.Quit()
