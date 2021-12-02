import importlib.metadata

__version__ = importlib.metadata.version("d2vg")

from .d2vg import main

from . import d2vg
from . import model_loaders
from . import parsers
from . import d2vg_setup_model
from . import types

setup_model_main = d2vg_setup_model.main
