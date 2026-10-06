"""One-page sample contract designed to be photographed for the camera demo.

Large, high-contrast text so phone OCR reads it reliably. Fictional gym; mixes fair and risky clauses.
Run:  .venv\\Scripts\\python -m samples.make_camera_page
Output: samples/camera_test_page.pdf and samples/camera_test_page.png
"""

from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent

TITLE = "FitZone Gym - Membership Terms"
CLAUSES = [
    "1. Your membership starts on the date you sign up and gives you access to all branches in Pune.",
    "2. The gym is open from 6 am to 10 pm on all days except public holidays.",
    "3. Your membership will automatically renew every month and your card will be charged unless you cancel 30 days in advance.",
    "4. FitZone may change these terms or the membership fee at any time at its sole discretion.",
    "5. FitZone may terminate your membership at any time without notice and no refund will be given.",
    "6. FitZone shall not be liable for any injury, loss or theft of personal belongings on the premises.",
    "7. A late fee of Rs. 200 per day will be charged on unpaid dues.",
    "8. Any dispute shall be resolved only by arbitration and you waive your right to go to court.",
    "9. Please bring a clean towel and indoor shoes for every workout.",
]


def main() -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 in points
    page.insert_textbox(pymupdf.Rect(50, 50, 545, 100), TITLE, fontsize=20, fontname="hebo")
    page.draw_line((50, 98), (545, 98), color=(0.75, 0.78, 0.82), width=1)
    y = 118
    for clause in CLAUSES:
        rect = pymupdf.Rect(50, y, 545, y + 80)
        spare = page.insert_textbox(rect, clause, fontsize=14, fontname="helv", lineheight=1.35)
        y += 80 - spare + 14  # move below the text actually used, plus a gap
    page.insert_textbox(pymupdf.Rect(50, 800, 545, 830), "Sample document for the ClauseGuard camera demo (fictional).",
                        fontsize=9, fontname="helv", color=(0.45, 0.48, 0.52))
    doc.save(OUT / "camera_test_page.pdf")
    page.get_pixmap(dpi=150).save(OUT / "camera_test_page.png")
    print("written:", OUT / "camera_test_page.pdf", "and", OUT / "camera_test_page.png")


if __name__ == "__main__":
    main()
