from typing import *
import sys


_ANSI_ESCAPE_CLEAR_CUR_LINE = "\x1b[1K\x1b[1G"


class ESession:
    def __init__(self, active=True):
        self._showing_flash_message = False
        self._activated = not not active
        # When self._activated is False, self._showing_flash_message must be False.
        # (Thus, when self._showing_flash_message is True, self._activated is True.)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def close(self):
        if self._showing_flash_message:
            print(_ANSI_ESCAPE_CLEAR_CUR_LINE, file=sys.stderr, end="", flush=True)
            self._showing_flash_message = False

    def __bool__(self) -> bool:
        return self.is_active()

    def is_active(self) -> bool:
        return self._activated

    def activate(self, state: bool) -> None:
        state = not not state
        if self._activated == state:
            return
        if state:
            self._activated = True
            return
        if self._showing_flash_message:
            print(_ANSI_ESCAPE_CLEAR_CUR_LINE, file=sys.stderr, end="", flush=True)
            self._showing_flash_message = False
        self._activated = False

    def flash(self, message: str) -> None:
        if not self._activated:
            return
        if self._showing_flash_message:
            print(_ANSI_ESCAPE_CLEAR_CUR_LINE, file=sys.stderr, end="")
        print(message, file=sys.stderr, end="", flush=True)
        self._showing_flash_message = True

    def flash_eval(self, message_callback: Callable[[], str]):
        if not self._activated:
            return
        if self._showing_flash_message:
            print(_ANSI_ESCAPE_CLEAR_CUR_LINE, file=sys.stderr, end="")
        print(message_callback(), file=sys.stderr, end="", flush=True)
        self._showing_flash_message = True

    def print(self, message: str, force: bool = False):
        if not force and not self._activated:
            return
        if self._showing_flash_message:
            print(_ANSI_ESCAPE_CLEAR_CUR_LINE, file=sys.stderr, end="")
        print(message, file=sys.stderr, flush=True)
        self._showing_flash_message = False

    def print_eval(self, message_callback: Callable[[], str], force: bool = False):
        if not force and not self._activated:
            return
        if self._showing_flash_message:
            print(_ANSI_ESCAPE_CLEAR_CUR_LINE, file=sys.stderr, end="")
        print(message_callback(), file=sys.stderr, flush=True)
        self._showing_flash_message = False


if __name__ == "__main__":
    from time import sleep

    with ESession() as essesion:
        essesion.print("This sample shows the usage of ESession.")
        sleep(1)
        essesion.print("still() method shows a message to stderr.")
        sleep(1)
        essesion.flash("flash() method also shows a message to stderr,")
        sleep(2)
        essesion.flash("but the message will not stay forever,")
        sleep(2)
        essesion.flash("and overwritten by another call of flush() or still(),")
        sleep(2)
        essesion.flash("or will be removed when leaving the session...")
        sleep(2)
        essesion.print("You saw all flash'ed messages gone.")
