from typing import *


T = TypeVar("T")


def remove_non_first_appearances(lst: Iterable[T]) -> List[T]:
    s = set()
    r = []
    for i in lst:
        if i not in s:
            r.append(i)
            s.add(i)
    return r


def concatinated(lists: Iterable[Iterable[T]]) -> List[T]:
    r = []
    for l in lists:
        r.extend(l)
    return r


def split_to_length(it: Iterable[T], group_size: int) -> Iterator[List[T]]:
    g = []
    for item in it:
        g.append(item)
        if len(g) == group_size:
            yield g
            g = []
    if len(g) > 0:
        yield g
