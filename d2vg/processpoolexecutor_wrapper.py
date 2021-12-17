import concurrent.futures
import os
import psutil


# ref: https://psutil.readthedocs.io/en/latest/index.html?highlight=Process#kill-process-tree
def kill_all_subprocesses():
    for child in psutil.Process(os.getpid()).children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


class ProcessPoolExecutor:
    def __init__(self, max_workers=None):
        if max_workers is not None and max_workers > 0:
            self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            self.map = self._executor.map
        else:
            self._executor = None
            self.map = map

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._executor is not None:
            self._executor.__exit__(exc_type, exc_value, traceback)

    def shutdown(self, wait=True, cancel_futures=False):
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            if cancel_futures:
                kill_all_subprocesses()  # might be better to call shutdown(wait=False, cancel_futures=True), in Python 3.10+
