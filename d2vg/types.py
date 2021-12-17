from typing import NewType, get_type_hints
from enum import Enum

import numpy as np


Vec = NewType("Vec", np.ndarray)
