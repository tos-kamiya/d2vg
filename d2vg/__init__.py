import importlib.metadata

__version__ = importlib.metadata.version("d2vg")

from .main import main

from . import model_loader
from . import parsers
from . import d2vg_setup_model
from . import vec
from . import index_db
from . import fnmatcher
from . import iter_funcs
from . import processpoolexecutor_wrapper
from . import raw_db
from . import cli
from . import embedding_utils
from . import search_result
from . import do_incremental_search
from . import do_index_search
from . import do_index_management
from . import file_opener


setup_model_main = d2vg_setup_model.main
