from typing import Iterable, Iterator, List, Tuple, TypeVar


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


def ranges_overwrapping(range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
    assert min(range1) >= 0
    assert min(range2) >= 0
    if 0 <= range1[0] <= range2[0]:
        return range2[0] < range1[1]
    else:
        # assert range2[0] <= range1[0]
        return range1[0] < range2[1]
