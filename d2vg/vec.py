from typing import List, NewType

import numpy as np
from gensim.matutils import unitvec


Vec = NewType("Vec", np.ndarray)


def to_float_list(vec: Vec) -> List[float]:
    return [float(d) for d in vec]


def inner_product_u(dv: Vec, pv: Vec) -> float:
    return float(np.inner(unitvec(dv), pv))


def inner_product_n(dv: Vec, pv: Vec) -> float:
    return float(np.inner(dv, pv))


def concatenate(vecs: List[Vec]) -> Vec:
    return np.concatenate(vecs, axis=0)
