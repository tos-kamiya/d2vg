from typing import *

from .iter_funcs import *

from .index_db import FileSignature
from .embedding_utils import extract_headline
from .vec import Vec


IPSRLL = Tuple[float, Tuple[int, int], List[str], List[List[str]]]
IPSRLL_OPT = Tuple[float, Tuple[int, int], Optional[List[str]], Optional[List[List[str]]]]


def prune_by_keywords(ip_srlls: Iterable[IPSRLL], keyword_set: FrozenSet[str], min_ip: Optional[float] = None) -> List[IPSRLL]:
    ipsrll_inc_kws: List[IPSRLL] = []
    lines: Optional[List[str]] = None
    line_tokens: Optional[List[List[str]]] = None
    for ip, (sp, ep), ls, lts in ip_srlls:
        if min_ip is not None and ip < min_ip:  # pruning by similarity (inner product)
            continue  # ip
        if line_tokens is None:
            assert ls is not None
            assert lts is not None
            lines = ls
            line_tokens = lts
        assert line_tokens is not None
        if not keyword_set.issubset(concatinated(line_tokens[sp:ep])):
            continue  # ip
        assert lines is not None
        ipsrll_inc_kws.append((ip, (sp, ep), lines, line_tokens))
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
    text_to_tokens: Callable[[str], List[str]],
    tokens_to_vec: Callable[[List[str]], Vec],
    pattern_vec: Vec,
    headline_length: int,
) -> None:
    for ipsrll, tf, _sig in search_results:
        ip, (b, e), lines, line_tokens = ipsrll
        if ip < 0:
            break  # for tf
        if lines is None:
            lines = parse(tf)
        lines = lines[b:e]
        if line_tokens:
            line_tokens = line_tokens[b:e]
        headline = extract_headline(lines, line_tokens, text_to_tokens, tokens_to_vec, pattern_vec, headline_length)
        print("%g\t%s:%d-%d\t%s" % (ip, tf, b + 1, e + 1, headline))
