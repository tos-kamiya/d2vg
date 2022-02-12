from typing import Generator, TextIO
from typing import TextIO

from contextlib import contextmanager


@contextmanager
def open_file(filename: str, mode: str = "r") -> Generator[TextIO, None, None]:
    with open(filename, mode, encoding="utf-8", errors="replace") as fp:
        yield fp
