from typing import Callable, FrozenSet, Iterable, List, Optional, Tuple

from .embedding_utils import extract_headline
from .index_db import FileSignature
from .iter_funcs import ranges_overwrapping
from .vec import Vec


IPSRLL = Tuple[float, Tuple[int, int], List[str]]
IPSRLL_OPT = Tuple[float, Tuple[int, int], Optional[List[str]]]


def prune_by_keywords(ip_srlls: Iterable[IPSRLL], keyword_set: FrozenSet[str], min_ip: Optional[float] = None) -> List[IPSRLL]:
    ipsrll_inc_kws: List[IPSRLL] = []
    for ipsrll in ip_srlls:
        ip, (b, e), ls = ipsrll
        if min_ip is not None and ip < min_ip:  # pruning by similarity (inner product)
            continue  # ip
        assert ls is not None
        remaining_keyword_set = set(keyword_set)
        for line in ls[b:e]:
            for k in list(remaining_keyword_set):
                if k in line:
                    remaining_keyword_set.discard(k)
            if not remaining_keyword_set:
                break  # for line
        if remaining_keyword_set:
            continue  # ip
        ipsrll_inc_kws.append(ipsrll)
    return ipsrll_inc_kws


def prune_overlapped_paragraphs(ip_srlls: List[IPSRLL_OPT], paragraph: bool) -> List[IPSRLL_OPT]:
    if not ip_srlls:
        return ip_srlls
    elif paragraph:
        dropped_index_set = set()
        for i, (ip_srll1, ip_srll2) in enumerate(zip(ip_srlls, ip_srlls[1:])):
            ip1, sr1 = ip_srll1[0], ip_srll1[1]
            ip2, sr2 = ip_srll2[0], ip_srll2[1]
            if ranges_overwrapping(sr1, sr2):
                if ip1 < ip2:
                    dropped_index_set.add(i)
                else:
                    dropped_index_set.add(i + 1)
        return [ip_srll for i, ip_srll in enumerate(ip_srlls) if i not in dropped_index_set]
    else:
        return [sorted(ip_srlls).pop()]  # take last (having the largest ip) item


SearchResult = Tuple[IPSRLL_OPT, str, FileSignature]


def print_search_results(
    search_results: List[SearchResult],
    parse: Callable[[str], List[str]],
    lines_to_vec: Callable[[List[str]], Vec],
    pattern_vec: Vec,
    headline_length: int,
    unit_vector: bool,
) -> None:
    for ipsrll, tf, _sig in search_results:
        ip, (b, e), lines = ipsrll
        if ip < 0:
            break  # for tf
        if lines is None:
            lines = parse(tf)
        lines = lines[b:e]
        headline = extract_headline(lines, lines_to_vec, pattern_vec, headline_length, unit_vector)
        print("%g\t%s:%d-%d\t%s" % (ip, tf, b + 1, e + 1, headline))
