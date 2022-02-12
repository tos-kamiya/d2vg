from typing import Callable, FrozenSet, Iterable, List, Optional, Tuple

from .embedding_utils import extract_headline
from .index_db import FileSignature
from .iter_funcs import ranges_overwrapping
from .vec import Vec


IPSRLS = Tuple[float, Tuple[int, int], List[str]]
IPSRLS_OPT = Tuple[float, Tuple[int, int], Optional[List[str]]]


def prune_overlapped_paragraphs(ipsrlss: List[IPSRLS_OPT], paragraph: bool) -> List[IPSRLS_OPT]:
    if not ipsrlss:
        return ipsrlss
    elif paragraph:
        dropped_index_set = set()
        for i, (ipsrls1, ipsrls2) in enumerate(zip(ipsrlss, ipsrlss[1:])):
            ip1, sr1 = ipsrls1[0], ipsrls1[1]
            ip2, sr2 = ipsrls2[0], ipsrls2[1]
            if ranges_overwrapping(sr1, sr2):
                if ip1 < ip2:
                    dropped_index_set.add(i)
                else:
                    dropped_index_set.add(i + 1)
        return [ipsrls for i, ipsrls in enumerate(ipsrlss) if i not in dropped_index_set]
    else:
        return [sorted(ipsrlss).pop()]  # take last (having the largest ip) item


SearchResult = Tuple[IPSRLS_OPT, str, FileSignature]


def print_search_results(
    search_results: List[SearchResult],
    parse: Callable[[str], List[str]],
    lines_to_vec: Callable[[List[str]], Vec],
    pattern_vec: Vec,
    headline_length: int,
    unit_vector: bool,
) -> None:
    for ipsrls, tf, _sig in search_results:
        ip, (b, e), lines = ipsrls
        if ip < 0:
            break  # for tf
        if lines is None:
            lines = parse(tf)
        lines = lines[b:e]
        headline = extract_headline(lines, lines_to_vec, pattern_vec, headline_length, unit_vector)
        print("%g\t%s:%d-%d\t%s" % (ip, tf, b + 1, e + 1, headline))
