import unittest

from pathlib import Path
import re
import tempfile

import d2vg


class ParserTest(unittest.TestCase):
    def test_text_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            p = Path(tempdir) / "a.txt"
            content = "1st line.\n2nd line.\n"
            p.write_text(content)
            read_content = d2vg.parsers.read_text_file(str(p))
            self.assertEqual(read_content, content)

    def test_html_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            p = Path(tempdir) / "a.html"
            content = """<!DOCTYPE html>
<html>
<body>
<p>1st paragraph.</p>
<p>2nd paragraph.</p>
</body>
</html>"""
            p.write_text(content)
            read_content = d2vg.parsers.html_parse(str(p))
            read_content = re.sub(r"\n+", r"\n", read_content).rstrip()
            self.assertEqual(read_content, "html\n1st paragraph.\n2nd paragraph.")

    def test_pdf_file(self):
        from borb.pdf.canvas.layout.page_layout.multi_column_layout import (
            SingleColumnLayout,
        )
        from borb.pdf.canvas.layout.text.paragraph import Paragraph
        from borb.pdf.document import Document
        from borb.pdf.page.page import Page
        from borb.pdf.pdf import PDF

        with tempfile.TemporaryDirectory() as tempdir:
            p = Path(tempdir) / "a.pdf"

            pdf = Document()
            page = Page()
            pdf.append_page(page)
            layout = SingleColumnLayout(page)
            layout.add(Paragraph("1st paragraph."))
            layout.add(Paragraph("2nd paragraph."))
            with open(p, "wb") as pdf_file_handle:
                PDF.dumps(pdf_file_handle, pdf)

            read_content = d2vg.parsers.pdf_parse(str(p))
            read_content = re.sub(r"\n+", r"\n", read_content).rstrip()
            self.assertEqual(read_content, "1st paragraph.\n2nd paragraph.")

    # !! not working !! ref: https://stackoverflow.com/questions/58186869/how-to-fix-the-bug-modulenotfounderror-no-module-named-exceptions-when-impo
    # def test_docx_file(self):
    #     from docx import Document

    #     with tempfile.TemporaryDirectory() as tempdir:
    #         p = Path(tempdir) / 'a.docx'

    #         document = Document()
    #         document.add_paragraph("1st paragraph.")
    #         document.add_paragraph("1st paragraph.")
    #         document.save(str(p))

    #         read_content = d2vg.parsers.docx_parse(str(p))
    #         read_content = re.sub(r'\n+', r'\n', read_content).rstrip()
    #         self.assertEqual(read_content, '1st paragraph.\n2nd paragraph.')


if __name__ == "__main__":
    unittest.main()
