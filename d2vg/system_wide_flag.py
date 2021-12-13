import os
import tempfile


_system_temp_dir = tempfile.gettempdir()


class SystemWideFlag:
    def __init__(self, value: bytes):
        fn = "%d.txt" % int.from_bytes(os.urandom(5), byteorder="little")
        self._tempfile = os.path.join(_system_temp_dir, fn)
        self._valid = True
        self.set(value)

    def set(self, value: bytes) -> None:
        assert self._valid
        with open(self._tempfile, "wb") as outp:
            outp.write(value)

    def get(self) -> bytes:
        assert self._valid
        with open(self._tempfile, "rb") as inp:
            return inp.read()

    def cleanup(self):
        assert self._valid
        os.remove(self._tempfile)
