# Copyright 2026 Christophe Le Douarec
"""Tests for the frames of a PDF page: as PDFium places it, and upright."""

from typing import TYPE_CHECKING

import pytest

from documentation_checker.extract.pdf.reader import open_pdf, read_page
from documentation_checker.images.frames import (
    from_upright,
    to_upright,
    upright_rotation,
    upright_size,
)
from tests.pdf_builders import TEXT_TURNS, build_moved_turned_pdf

if TYPE_CHECKING:
    from pathlib import Path

# A crop box away from the origin, 300 by 400 points, and a box on it, 30,
# 40, 90 and 70 points from the corner of the crop box.
CROP = (50.0, 60.0, 350.0, 460.0)
BOX = (80.0, 100.0, 140.0, 130.0)
TURNS = [
    pytest.param(matrix, rotation, id=name)
    for matrix, rotation, name in TEXT_TURNS
]


@pytest.mark.parametrize("turn", range(4))
def test_from_upright_reverses_to_upright(turn: int) -> None:
    """A box turned upright and back is the same box."""
    upright = to_upright(BOX, CROP, turn)
    assert from_upright(upright, CROP, turn) == pytest.approx(BOX)


@pytest.mark.parametrize(
    ("turn", "upright"),
    [
        (0, (30, 40, 90, 70)),
        (1, (330, 30, 360, 90)),
        (2, (210, 330, 270, 360)),
        (3, (40, 210, 70, 270)),
    ],
)
def test_to_upright_turns_from_the_crop_corner(
    turn: int,
    upright: tuple[float, float, float, float],
) -> None:
    """A box upright is measured from the corner of the crop box turned."""
    assert to_upright(BOX, CROP, turn) == pytest.approx(upright)


@pytest.mark.parametrize("turn", range(4))
def test_a_box_upright_lies_within_the_page_upright(turn: int) -> None:
    """The edges of a box upright are in order and inside the page."""
    x0, y0, x1, y1 = to_upright(BOX, CROP, turn)
    width, height = upright_size(CROP, turn)
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


@pytest.mark.parametrize(
    ("turn", "size"),
    [(0, (300, 400)), (1, (400, 300)), (2, (300, 400)), (3, (400, 300))],
)
def test_upright_size_swaps_for_a_quarter_turn(
    turn: int,
    size: tuple[float, float],
) -> None:
    """An odd number of quarter turns swaps the width and the height."""
    assert upright_size(CROP, turn) == size


@pytest.mark.parametrize(
    ("turn", "rotation", "added"),
    [
        (0, 0, 0),
        (1, 270, 0),
        (2, 180, 0),
        (3, 90, 0),
        (0, 90, 270),
        (1, 0, 270),
        (3, 0, 90),
        (2, 0, 180),
        (0, 270, 90),
    ],
)
def test_upright_rotation_completes_the_rotation_of_the_page(
    turn: int,
    rotation: int,
    added: int,
) -> None:
    """The added rotation and that of the page turn the text upright."""
    assert upright_rotation(turn, rotation) == added


@pytest.mark.parametrize(("matrix", "rotation"), TURNS)
def test_frames_agree_with_the_reader(
    tmp_path: Path,
    matrix: str,
    rotation: int,
) -> None:
    """A box of a character of PDFium, upright, is where the reader has it."""
    path = build_moved_turned_pdf(
        tmp_path / "turned.pdf",
        matrix,
        ["First line"],
        rotation=rotation,
    )
    with open_pdf(path) as pdf:
        page = read_page(pdf, 0)
        pdf_page = pdf[0]
        crop = pdf_page.get_bbox()
        textpage = pdf_page.get_textpage()
        box = textpage.get_charbox(0)
        textpage.close()
        pdf_page.close()
    first = page.chars[0]
    x0, y0, _, y1 = to_upright(box, crop, page.turn)
    assert x0 == pytest.approx(first.x0, abs=0.01)
    assert y0 == pytest.approx(first.baseline, abs=0.01)
    assert y1 > first.baseline
    assert upright_rotation(page.turn, rotation) == 0
