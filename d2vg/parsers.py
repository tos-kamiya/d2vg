from io import StringIO
import re

import bs4

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

import docx2txt


class PraseError(Exception):
    pass


def parse(file_name):
    try:
        text = _parse_i(file_name)
    except Exception as e:
        raise PraseError("ParseError: in parsing file: %s" % repr(file_name)) from e

    if text is not None:
        lines = text.split('\n')
        r = []
        for L in lines:
            L = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', L)
            L = re.sub(r'\s+', ' ', L)
            r.append(L)
        return r
    return None


def _parse_i(file_name):
    i = file_name.rfind('.')
    if i < 0:
        return None
    extension = file_name[i:].lower()
    if extension in ['.html', 'htm']:
        return html_parse(file_name)
    elif extension == '.pdf':
        return pdf_parse(file_name)
    elif extension == '.docx':
        return docx_parse(file_name)
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
        text = re.sub(r'(cid:\d+)', '', text)  # remove unknown glyphs

    return text


def html_parse(file_name):
    with open(file_name) as inp:
        html_doc = inp.read()
        soup = bs4.BeautifulSoup(html_doc, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        texts = soup.find_all(text=True)
    return '\n'.join(texts)


def docx_parse(file_name):
    return docx2txt.process(file_name)