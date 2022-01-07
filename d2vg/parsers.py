from typing import *

import os
import platform
import re
import subprocess
import sys
import zipfile

import bs4
import docx2txt

if platform.system() != "Windows":
    import pdftotext

from .file_opener import open_file


_script_dir: str = os.path.dirname(os.path.realpath(__file__))

_ja_nkf_abspath: Optional[str] = None
if platform.system() == "Windows" and os.path.exists(os.path.join(_script_dir, "nkf32.exe")):
    _ja_nkf_abspath = os.path.abspath(os.path.join(_script_dir, "nkf32.exe"))


if _ja_nkf_abspath:

    def read_text_file(file_name: str) -> str:
        b = subprocess.check_output([_ja_nkf_abspath, "-Lu", "--oc=UTF-8", file_name])
        return b.decode("utf-8")


else:

    def read_text_file(file_name: str) -> str:
        with open_file(file_name) as inp:
            return inp.read()


class ParseError(Exception):
    pass


class Parser:
    def parse(self, file_name: str) -> List[str]:
        try:
            text = self._parse_i(file_name)
        except ParseError as e:
            raise e
        except Exception as e:
            raise ParseError("ParseError: in parsing file: %s" % repr(file_name)) from e
        return self.clean_text(text)

    def parse_text(self, text: str) -> List[str]:
        return self.clean_text(text)

    def clean_text(self, text: str) -> List[str]:
        lines = text.split("\n")
        r = []
        for L in lines:
            L = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", L)
            L = re.sub(r"\s+", " ", L)
            r.append(L)
        return r

    def _parse_i(self, file_name: str) -> str:
        assert file_name != "-"

        i = file_name.rfind(".")
        if i < 0:
            raise ParseError("ParseError: file has NO extension: %s" % repr(file_name))

        extension = file_name[i:].lower()

        if extension in [".html", "htm"]:
            return html_parse(file_name)
        elif extension == ".pdf":
            return pdf_parse(file_name)
        elif extension == ".docx":
            return docx_parse(file_name)
        else:
            return read_text_file(file_name)


if platform.system() != "Windows":

    def pdf_parse(file_name: str) -> str:
        try:
            with open(file_name, "rb") as f:
                pdf = pdftotext.PDF(f)
        except pdftotext.Error as e:
            raise ParseError("ParseError: %s, file: %s" % (str(e), repr(file_name)))

        page_texts = [page for page in pdf]
        text = "".join(page_texts)
        # text = re.sub(r'(cid:\d+)', '', text)  # remove unknown glyphs

        return text


else:
    import tempfile

    _system_temp_dir = tempfile.gettempdir()

    def pdf_parse(file_name: str) -> str:
        tempf = os.path.join(
            _system_temp_dir,
            "%d.txt" % int.from_bytes(os.urandom(5), byteorder="little"),
        )
        try:
            cmd = ["pdftotext.exe", file_name, tempf]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode != 0:
                raise ParseError("ParseError: %s, file: %s" % (p.stderr.decode("utf-8").rstrip(), repr(file_name)))
            with open_file(tempf) as f:
                text = f.read()
        finally:
            if os.path.exists(tempf):
                os.remove(tempf)
        return text


def html_parse(file_name: str) -> str:
    with open_file(file_name) as inp:
        html_doc = inp.read()
        soup = bs4.BeautifulSoup(html_doc, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        texts = soup.find_all(text=True)
    return "\n".join(texts)


def docx_parse(file_name: str) -> str:
    try:
        return docx2txt.process(file_name)
    except zipfile.BadZipFile as _e:
        raise ParseError("ParseError: encrypted or corrupted .docx file: %s" % repr(file_name))
