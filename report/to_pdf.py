"""Export a built .docx to PDF and report its pagination.

    python report/to_pdf.py report/assignment2_submission.docx

DispatchEx forces a private Word process, so a document already open in the user's
session is never touched. Prints the body page count (Introduction -> References),
which is the figure the 10-page cap applies to.
"""

import sys
from pathlib import Path

import win32com.client as win32

WD_PAGES = 2            # wdStatisticPages
WD_END_PAGE = 3         # wdActiveEndPageNumber
WD_EXPORT_PDF = 17      # wdExportFormatPDF
WD_EXPORT_DOC = 0       # wdExportDocumentContent

src = Path(sys.argv[1]).resolve()
dst = src.with_suffix(".pdf")

word = win32.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = False
try:
    doc = word.Documents.Open(str(src), ReadOnly=False, AddToRecentFiles=False)
    total = doc.ComputeStatistics(WD_PAGES)   # paginates as a side effect

    appendix_page = None
    for p in doc.Paragraphs:
        if p.Range.Text.strip() == "Appendix":
            appendix_page = p.Range.Information(WD_END_PAGE)
            break

    doc.ExportAsFixedFormat(
        OutputFileName=str(dst),
        ExportFormat=WD_EXPORT_PDF,
        OpenAfterExport=False,
        Item=WD_EXPORT_DOC,
    )
    doc.Close(False)
finally:
    word.Quit()

print(f"wrote {dst}")
print(f"total pages:     {total}")
if appendix_page:
    print(f"appendix starts: p{appendix_page}")
    print(f"CAPPED BODY:     {appendix_page - 1} pages (Introduction -> References)")
