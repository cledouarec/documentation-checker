# Copyright 2026 Christophe Le Douarec
"""Link the pages of two documents, as the alignment of their blocks tells.

Each pair of blocks that the two documents share, in their common order,
links the pages of its reference blocks with the pages of its new blocks, by
the characters of its reference blocks: the characters of a block are shared
out among the pages that it lies on, as the heights of its boxes there tell,
and those of each page of the reference among the pages of the new blocks,
in the same way. The reports match the pages one to one with
`heaviest_chain`, in the order of both documents, so that the links of the
pages matched weigh the most, and show the pages so matched side by side.
"""

from collections import Counter
from typing import TYPE_CHECKING

from documentation_checker.compare.drafts import MATCHED

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from documentation_checker.compare.align import Pair
    from documentation_checker.model.document import Block


def _chars_by_page(blocks: Iterable[Block]) -> Counter[int]:
    """Return the characters of some blocks on each page, by its number.

    The characters of a block are shared out among the pages of its boxes,
    as their heights there tell; a block without boxes, as the blocks of a
    run without screenshots, is all on the page of its source, if it has
    one.
    """
    chars: Counter[int] = Counter()
    for block in blocks:
        heights: Counter[int] = Counter()
        for box in block.source.boxes:
            heights[box.page + 1] += max(box.top - box.bottom, 0.0)
        total = heights.total()
        if total > 0:
            for page, height in heights.items():
                chars[page] += len(block.text) * height / total
        elif block.source.page is not None:
            chars[block.source.page] += len(block.text)
    return chars


def page_links(
    ref: Sequence[Block],
    new: Sequence[Block],
    pairs: Sequence[Pair],
) -> Counter[tuple[int, int]]:
    """Return how much text links each page of the reference with each new one.

    Args:
        ref: The compared blocks of the reference.
        new: The compared blocks of the new document.
        pairs: Their alignment.

    Returns:
        For each page of the reference and page of the new document, by
        their numbers from 1, the characters of the reference blocks of the
        pairs of shared blocks, in the common order, on the page of the
        reference, shared out among the pages of the new blocks as their
        characters are. Pages that no such pair links, and blocks without a
        page, as those of HTML documents, are left out.
    """
    links: Counter[tuple[int, int]] = Counter()
    for pair in pairs:
        if pair.kind not in MATCHED or pair.moved:
            continue
        ref_chars = _chars_by_page(ref[i] for i in pair.ref)
        new_chars = _chars_by_page(new[i] for i in pair.new)
        new_total = new_chars.total()
        if not new_total:
            continue
        for ref_page, chars in ref_chars.items():
            for new_page, share in new_chars.items():
                links[ref_page, new_page] += chars * share / new_total
    return links


def heaviest_chain(
    weights: Mapping[tuple[int, int], float],
) -> tuple[tuple[int, int], ...]:
    """Return the pairs of places that rise together and weigh the most.

    The heaviest chain is found among the pairs taken by their first place,
    a tree of Fenwick over their second places giving the heaviest chain
    that ends before each of them.

    Args:
        weights: The weight, above 0, of each pair of a place in one order
            and a place in another.

    Returns:
        Pairs of `weights` whose two places both rise from one pair to the
        next, so that each place is in one pair at most, and whose weights
        add up to the most, in that order.
    """
    # The pairs by their first place, and for one first place by their
    # second place falling, so that two of them never follow each other.
    items = sorted(weights, key=lambda pair: (pair[0], -pair[1]))
    seconds = sorted({second for _, second in items})
    ranks = {second: rank for rank, second in enumerate(seconds, 1)}
    # For each rank, the heaviest chain that ends there, and its last pair.
    tree: list[tuple[float, int]] = [(0.0, -1)] * (len(seconds) + 1)
    totals = [0.0] * len(items)
    before = [-1] * len(items)
    for position, pair in enumerate(items):
        best = (0.0, -1)
        rank = ranks[pair[1]] - 1
        while rank:
            best = max(best, tree[rank])
            rank &= rank - 1
        totals[position] = best[0] + weights[pair]
        before[position] = best[1]
        rank = ranks[pair[1]]
        while rank < len(tree):
            tree[rank] = max(tree[rank], (totals[position], position))
            rank += rank & -rank
    last = max(range(len(items)), key=totals.__getitem__, default=-1)
    chain: list[tuple[int, int]] = []
    while last >= 0:
        chain.append(items[last])
        last = before[last]
    return tuple(reversed(chain))
