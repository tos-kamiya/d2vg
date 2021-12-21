from typing import *

from math import pow

import numpy as np
from gensim.matutils import unitvec


Vec = NewType("Vec", np.ndarray)


def to_float_list(vec: Vec) -> List[float]:
    return [float(d) for d in vec]


def normalize_vec(vec: Vec) -> Vec:
    veclen = np.linalg.norm(vec)
    if veclen == 0.0:
        return vec
    return pow(veclen, -0.5) * vec


def inner_product_u(dv: Vec, pv: Vec) -> float:
    return float(np.inner(unitvec(dv), pv))


def inner_product_n(dv: Vec, pv: Vec) -> float:
    return float(np.inner(normalize_vec(dv), pv))


