# Copyright 2026 Christophe Le Douarec
"""Tests for the links between the pages of two compared documents."""

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from documentation_checker.compare.align import Pair, PairKind
from documentation_checker.compare.pages import heaviest_chain, page_links
from documentation_checker.config import Config
from documentation_checker.model.document import PageBox, SourceRef
from documentation_checker.pipeline import run_compare
from tests.builders import make_block
from tests.pdf_builders import build_lines_pdf

if TYPE_CHECKING:
    from pathlib import Path

    from documentation_checker.model.document import Block


def _block(
    text: str,
    page: int | None,
    *heights: tuple[int, float],
) -> Block:
    """Return a paragraph of a page, or of no page for None.

    Each height is that of a box of the block on a page, by its position.
    """
    boxes = tuple(
        PageBox(position, 0.0, 0.0, 100.0, height)
        for position, height in heights
    )
    return replace(
        make_block(text),
        source=SourceRef("doc.pdf", page=page, boxes=boxes),
    )


def test_the_heaviest_chain_rises_in_both_orders() -> None:
    """Of pairs that cross, or share a place, the heaviest ones are kept."""
    assert heaviest_chain({(1, 1): 5, (2, 2): 5, (3, 3): 5}) == (
        (1, 1),
        (2, 2),
        (3, 3),
    )
    assert heaviest_chain({(1, 2): 10, (2, 1): 3, (3, 3): 4}) == (
        (1, 2),
        (3, 3),
    )
    assert heaviest_chain({(1, 1): 5, (1, 2): 6, (2, 2): 2}) == (
        (1, 1),
        (2, 2),
    )
    assert heaviest_chain({}) == ()


def test_shared_blocks_link_their_pages_by_their_text() -> None:
    """Shared blocks in order link pages; moved and lost ones do not."""
    ref = [
        _block("one two", 1),
        _block("three", 2),
        _block("lost", 2),
        _block("moved", 3),
        _block("html", None),
    ]
    new = [
        _block("one two", 1),
        _block("three!", 1),
        _block("moved", 2),
        _block("html", None),
    ]
    pairs = [
        Pair(PairKind.EQUAL, (0,), (0,)),
        Pair(PairKind.MODIFIED, (1,), (1,)),
        Pair(PairKind.DELETED, (2,), ()),
        Pair(PairKind.EQUAL, (3,), (2,), moved=True),
        Pair(PairKind.EQUAL, (4,), (3,)),
    ]
    assert page_links(ref, new, pairs) == {(1, 1): 7, (2, 1): 5}


def test_a_block_over_pages_links_them_by_the_heights_of_its_boxes() -> None:
    """The characters of a block are shared out as its boxes are high."""
    ref = [_block("x" * 80, 1, (0, 30.0), (1, 10.0))]
    new = [_block("x" * 80, 2, (1, 10.0), (2, 30.0))]
    links = page_links(ref, new, [Pair(PairKind.EQUAL, (0,), (0,))])
    assert links == pytest.approx(
        {(1, 2): 15.0, (1, 3): 45.0, (2, 2): 5.0, (2, 3): 15.0},
    )
    # Most of the block is on the first page of the reference and on the
    # third page of the new document.
    assert heaviest_chain(links) == ((1, 3),)


@pytest.mark.parametrize(
    ("screenshots", "links"),
    [(True, {(1, 1), (1, 2), (2, 1), (2, 2)}), (False, {(1, 1)})],
    ids=["boxes", "pages"],
)
def test_a_comparison_links_its_pages(
    tmp_path: Path,
    *,
    screenshots: bool,
    links: set[tuple[int, int]],
) -> None:
    """A paragraph over two pages links them, if the boxes tell where."""
    lines = [f"line {i} of the long paragraph" for i in range(100)]
    ref = build_lines_pdf([lines], tmp_path / "ref.pdf")
    new = build_lines_pdf([lines], tmp_path / "new.pdf")
    result = run_compare(ref, new, Config(screenshots=screenshots))
    assert set(result.page_links) == links
    assert heaviest_chain(result.page_links)[0] == (1, 1)
