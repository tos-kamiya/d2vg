from io import StringIO
import os
import re
import subprocess
import sys

import bs4

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

import docx2txt


_script_dir = os.path.dirname(os.path.realpath(__file__))

_ja_nkf_abspath = None
if os.name == 'nt' and os.path.exists(os.path.join(_script_dir, 'nkf32.exe')):
    _ja_nkf_abspath = os.path.abspath(os.path.join(_script_dir, 'nkf32.exe'))


if _ja_nkf_abspath:
    def read_text_file(file_name):
        b = subprocess.check_output([_ja_nkf_abspath, "-Lu", "--oc=UTF-8", file_name])
        return b.decode('utf-8')
else:
    def read_text_file(file_name):
        with open(file_name) as inp:
            return inp.read()


class PraseError(Exception):
    pass


class Parser:
    def __init__(self):
        self.__stdin_text = None

    def parse(self, file_name):
        try:
            text = self._parse_i(file_name)
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

    def _parse_i(self, file_name):
        if file_name == '-':
            if self.__stdin_text is None:
                self.__stdin_text = sys.stdin.read()
            return self.__stdin_text

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
            return read_text_file(file_name)


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
