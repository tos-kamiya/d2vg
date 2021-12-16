import importlib.metadata

__version__ = importlib.metadata.version("d2vg")

from .d2vg import main

from . import d2vg
from . import model_loader
from . import parsers
from . import d2vg_setup_model
from . import types
from . import index_db
from . import fnmatcher
from . import iter_funcs
from . import processpoolexecutor_wrapper
from . import raw_db

setup_model_main = d2vg_setup_model.main
