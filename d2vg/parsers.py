from typing import *

from io import StringIO
import os
import re
import subprocess
import sys

import bs4
import pdftotext
import docx2txt


_script_dir: str = os.path.dirname(os.path.realpath(__file__))

_ja_nkf_abspath: Union[str, None] = None
if os.name == 'nt' and os.path.exists(os.path.join(_script_dir, 'nkf32.exe')):
    _ja_nkf_abspath = os.path.abspath(os.path.join(_script_dir, 'nkf32.exe'))


if _ja_nkf_abspath:
    def read_text_file(file_name: str) -> str:
        b = subprocess.check_output([_ja_nkf_abspath, "-Lu", "--oc=UTF-8", file_name])
        return b.decode('utf-8')
else:
    def read_text_file(file_name: str) -> str:
        with open(file_name) as inp:
            return inp.read()


class PraseError(Exception):
    pass


class Parser:
    def __init__(self):
        self.__stdin_text = None

    def parse(self, file_name: str) -> List[str]:
        try:
            text = self._parse_i(file_name)
        except Exception as e:
            raise PraseError("ParseError: in parsing file: %s" % repr(file_name)) from e

        lines = text.split('\n')
        r = []
        for L in lines:
            L = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', L)
            L = re.sub(r'\s+', ' ', L)
            r.append(L)
        return r

    def _parse_i(self, file_name: str) -> str:
        if file_name == '-':
            if self.__stdin_text is None:
                self.__stdin_text = sys.stdin.read()
            return self.__stdin_text

        i = file_name.rfind('.')
        if i < 0:
            raise PraseError("ParseError: file has NO extension: %s" % repr(file_name))

        extension = file_name[i:].lower()

        if extension in ['.html', 'htm']:
            return html_parse(file_name)
        elif extension == '.pdf':
            return pdf_parse(file_name)
        elif extension == '.docx':
            return docx_parse(file_name)
        else:
            return read_text_file(file_name)


def pdf_parse(file_name: str) -> str:
    with open(file_name, "rb") as f:
        pdf = pdftotext.PDF(f)

    page_texts = [page for page in pdf]
    text = ''.join(page_texts)
    # text = re.sub(r'(cid:\d+)', '', text)  # remove unknown glyphs

    return text


def html_parse(file_name: str) -> str:
    with open(file_name) as inp:
        html_doc = inp.read()
        soup = bs4.BeautifulSoup(html_doc, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        texts = soup.find_all(text=True)
    return '\n'.join(texts)


def docx_parse(file_name: str) -> str:
    return docx2txt.process(file_name)
