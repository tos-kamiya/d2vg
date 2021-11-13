from io import StringIO
import re

import bs4
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


def parse(file_name):
    text = _parse_i(file_name)
    if text is not None:
        text = re.sub(r'[\x00-\x09\x0b-\x1f\x7f-\x9f]', '', text)
    return text


def _parse_i(file_name):
    i = file_name.rfind('.')
    if i < 0:
        return None
    extension = file_name[i:].lower()
    if extension in ['.html', 'htm']:
        return html_parse(file_name)
    elif extension == '.pdf':
        return pdf_parse(file_name)
    else:
        with open(file_name) as inp:
            text = inp.read()
        return text


def pdf_parse(file_name):
    manager = PDFResourceManager()

    with StringIO() as outp:
        with open(file_name, 'rb') as input:
            with TextConverter(manager, outp, codec='utf-8', laparams=LAParams()) as conv:
                interpreter = PDFPageInterpreter(manager, conv)
                for page in PDFPage.get_pages(input):
                    interpreter.process_page(page)
        text = outp.getvalue()

    return text


def html_parse(file_name):
    with open(file_name) as inp:
        html_doc = inp.read()
        soup = bs4.BeautifulSoup(html_doc, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        texts = soup.find_all(text=True)
    return '\n'.join(texts)
