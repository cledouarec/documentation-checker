# Copyright 2026 Christophe Le Douarec
"""Tests for the screenshots of the pages of PDF documents."""

import math
import os
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from PIL import Image
from reportlab.lib.pagesizes import A4

from documentation_checker.config import PdfConfig
from documentation_checker.extract.pdf.loader import load_pdf
from documentation_checker.images.load import RENDER_SIDE_MAX
from documentation_checker.images.locate import Wanted
from documentation_checker.images.pixels import open_raster
from documentation_checker.images.screenshot import (
    BORDERS,
    CONTEXT_ACROSS,
    CONTEXT_DOWN,
    PAGE_SCALE,
    SHOTS_PER_WORKER,
    ZOOM_MIN_WIDTH,
    ZOOM_SCALE,
    PdfShooter,
    ShotRequest,
    Shots,
    edge_row,
    encode_shot,
    quiet_rows,
    zoom_area,
)
from documentation_checker.model.document import PageBox
from tests.pdf_builders import (
    TEXT_TURNS,
    build_image_stream_pdf,
    build_lines_pdf,
    build_moved_turned_pdf,
    build_single_page_pdf,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from documentation_checker.model.document import Document

DARK = 100  # The pixels of the text are darker than this on each channel.
LINES = ["The first line of the text.", "Second line with a word."]
TURNS = [
    pytest.param(matrix, rotation, id=name)
    for matrix, rotation, name in TEXT_TURNS
]
BLOCK = (72.0, 400.0, 523.0, 700.0)  # A block of a page of A4.
MARK = (100.0, 590.0, 130.0, 602.0)  # A word of a line of that block.


def _load(path: Path) -> Document:
    """Load a PDF with the boxes of its blocks."""
    return load_pdf(path, PdfConfig(), lang="en", boxes=True)


def _found(
    image: Image.Image,
    color: tuple[int, int, int],
) -> tuple[int, int, int, int] | None:
    """Return the left, top, right and bottom pixels of a color.

    The borders and the frames are drawn last, in their colors, and without
    antialiasing.
    """
    pixels = np.asarray(image.convert("RGB"))
    rows, cols = np.nonzero((pixels == color).all(axis=-1))
    if not len(rows):
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def _dark(image: Image.Image) -> np.ndarray:
    """Return which pixels of an image are dark, as the text is."""
    return (np.asarray(image.convert("RGB")) < DARK).all(axis=-1)


def _dark_inside(image: Image.Image, box: tuple[int, int, int, int]) -> int:
    """Return the number of dark pixels strictly inside a box of pixels."""
    left, top, right, bottom = box
    inside = _dark(image)[top + 3 : bottom - 2, left + 3 : right - 2]
    return int(np.count_nonzero(inside))


def _page(path: Path, requests: Sequence[ShotRequest]) -> Image.Image | None:
    """Return the page of the first request, as `shoot_all` makes it."""
    with PdfShooter(path) as shooter:
        shot = shooter.shoot_all(requests).pages[0]
    return None if shot is None else open_raster(shot.image.data)


def test_the_zoom_shows_the_lines_of_its_marks_across_their_block() -> None:
    """The zoom holds the line of a word, as wide as its block."""
    assert zoom_area([BLOCK], [MARK], A4) == (
        BLOCK[0] - CONTEXT_ACROSS,
        MARK[1] - CONTEXT_DOWN,
        BLOCK[2] + CONTEXT_ACROSS,
        MARK[3] + CONTEXT_DOWN,
    )


def test_a_caret_zooms_on_its_line() -> None:
    """A caret, which has no width, shows its line across its block."""
    caret = (150.0, 590.0, 150.0, 602.0)
    assert zoom_area([BLOCK], [caret], A4) == zoom_area([BLOCK], [MARK], A4)


def test_only_the_blocks_that_hold_a_mark_set_the_width() -> None:
    """In two columns, the zoom spans the column of its mark, widened."""
    left_column = (72.0, 400.0, 290.0, 700.0)
    right_column = (305.0, 400.0, 523.0, 700.0)
    area = zoom_area([left_column, right_column], [MARK], A4)
    assert area == (31.0, 566.0, 331.0, 626.0)


def test_a_zoom_without_marks_shows_its_blocks() -> None:
    """A block tinted whole, or framed, is shown whole."""
    assert zoom_area([BLOCK], [], A4) == (60.0, 376.0, 535.0, 724.0)


def test_a_narrow_zoom_is_widened_around_its_middle() -> None:
    """A zoom has `ZOOM_MIN_WIDTH` points across at least."""
    area = zoom_area([(200.0, 400.0, 260.0, 420.0)], [], A4)
    assert area == (80.0, 376.0, 380.0, 444.0)


def test_a_widened_zoom_stays_on_the_page() -> None:
    """Near a side of the page, the widening goes the other way."""
    width = A4[0]
    area = zoom_area([(width - 80, 400.0, width - 20, 420.0)], [], A4)
    assert area is not None
    assert area[2] == width
    assert area[2] - area[0] == pytest.approx(ZOOM_MIN_WIDTH)


@pytest.mark.parametrize(
    "block",
    [
        (A4[0] + 20, 400.0, A4[0] + 80, 420.0),
        (72.0, -80.0, 300.0, -30.0),
    ],
    ids=["right", "under"],
)
def test_a_zoom_off_the_page_is_none(
    block: tuple[float, float, float, float],
) -> None:
    """Blocks away from the page, with their context, give no zoom."""
    assert zoom_area([block], [], A4) is None


def test_quiet_rows_have_no_ink_between_their_columns() -> None:
    """Ink out of the columns, or on every row, leaves a row quiet."""
    pixels = np.full((6, 10), 255, dtype=np.uint8)
    pixels[:, 2] = 0  # A rule down every row.
    pixels[1, 5] = 0  # Ink in the columns.
    pixels[3, 8] = 0  # Ink out of the columns.
    pixels[4, 6] = 200  # A light gray, which is no ink.
    image = Image.fromarray(pixels, "L").convert("RGBX")
    quiet = quiet_rows(image, (1, 7))
    assert quiet.tolist() == [True, False, True, True, True, True]


@pytest.mark.parametrize(
    ("row", "outward", "reach", "limit", "expected"),
    [
        (5, 1, 2, 0, 5),  # A quiet row stays.
        (2, 1, 2, 0, 4),  # Down, to the end of the line that it cut.
        (2, -1, 2, 9, 0),  # Up, to the end of the line that it cut.
        (8, 1, 2, 0, 6),  # Nothing quiet down: up, leaving the line out.
        (8, 1, 2, 7, 8),  # The limit keeps it from leaving the line out.
        (2, 1, 1, 0, 2),  # Nothing quiet within the reach.
    ],
    ids=["quiet", "down", "up", "inward", "limit", "reach"],
)
def test_an_edge_moves_to_the_nearest_quiet_row(
    row: int,
    outward: Literal[-1, 1],
    reach: int,
    limit: int,
    expected: int,
) -> None:
    """An edge goes outward to a quiet row, else inward, within its reach."""
    quiet = [True, False, False, False, True, True, True, False, False]
    moved = edge_row(quiet, row, outward=outward, reach=reach, limit=limit)
    assert moved == expected


def _turned(
    tmp_path: Path,
    matrix: str,
    rotation: int,
) -> tuple[Path, ShotRequest]:
    """Return a page of `TEXT_TURNS` and the request of a word of its text.

    The second line of the text has its baseline 238 points up the page
    turned upright, and the word starts 95 points from its left.
    """
    path = build_moved_turned_pdf(
        tmp_path / "turned.pdf",
        matrix,
        LINES,
        rotation=rotation,
    )
    (block,) = _load(path).blocks
    request = ShotRequest(
        block.source.boxes,
        "find",
        "delete",
        Wanted("Second line ", "with", " a word."),
    )
    return path, request


def _word_border(
    image: Image.Image,
    scale: float,
) -> tuple[int, int, int, int]:
    """Return the border of the word of `_turned`, checked, on a render.

    The word "with" is 4 letters of Helvetica 10, about 19 points, which
    start 95 points from the left of the page: from the left of the render.
    """
    border = _found(image, BORDERS["delete"])
    assert border is not None
    left, _, right, _ = border
    assert 15 * scale < right - left < 30 * scale
    assert _dark_inside(image, border) > 0
    assert left == pytest.approx(95 * scale, abs=3 * scale)
    return border


def _check_context(image: Image.Image, border: tuple[int, ...]) -> None:
    """Check that a zoom shows `CONTEXT_DOWN` points above and under a mark."""
    _, top, _, bottom = border
    context = CONTEXT_DOWN * ZOOM_SCALE
    assert top == pytest.approx(context, abs=3)
    assert image.height - bottom == pytest.approx(context, abs=3)


@pytest.mark.parametrize("rotated", [True, False], ids=["rotated", "sideways"])
@pytest.mark.parametrize(("matrix", "rotation"), TURNS)
def test_the_screenshots_are_upright_whatever_the_turn(
    tmp_path: Path,
    matrix: str,
    rotation: int,
    *,
    rotated: bool,
) -> None:
    """The page and the zoom are upright, and the mark borders its word.

    The page has the rotation that shows its text upright in a viewer, or
    none, its text then set sideways: the screenshots are upright either
    way. Widened to `ZOOM_MIN_WIDTH`, the zoom starts at the left of the
    page.
    """
    path, request = _turned(tmp_path, matrix, rotation if rotated else 0)
    page = _page(path, [request])
    with PdfShooter(path) as shooter:
        zoom = shooter.render(request)
    assert page is not None
    assert zoom is not None
    width, height = (300, 400) if rotation % 180 == 0 else (400, 300)
    assert page.size == (width * PAGE_SCALE, height * PAGE_SCALE)
    _, top, _, bottom = _word_border(page, PAGE_SCALE)
    assert bottom - top < 16 * PAGE_SCALE
    assert (top + bottom) / 2 == pytest.approx(
        (height - 238 - 3) * PAGE_SCALE,
        abs=4 * PAGE_SCALE,
    )
    image, index = zoom
    assert index == 0
    assert image.width == ZOOM_MIN_WIDTH * ZOOM_SCALE
    _check_context(image, _word_border(image, ZOOM_SCALE))


def test_the_zoom_shows_the_line_of_its_change(tmp_path: Path) -> None:
    """The zoom of a word of a long paragraph is its line and two around."""
    lines = [
        f"line {i} of a long paragraph that goes on and on, line after line,"
        " to the right"
        for i in range(40)
    ]
    path = build_lines_pdf([lines], tmp_path / "long.pdf")
    (block,) = _load(path).blocks
    request = ShotRequest(
        block.source.boxes,
        "find",
        "insert",
        Wanted("line 19 of a ", "long", " paragraph"),
    )
    with PdfShooter(path) as shooter:
        shot = shooter.render(request)
    assert shot is not None
    image, _ = shot
    (box,) = block.source.boxes
    across = box.x1 - box.x0 + 2 * CONTEXT_ACROSS
    assert across > ZOOM_MIN_WIDTH
    assert image.width == pytest.approx(across * ZOOM_SCALE, abs=2)
    border = _found(image, BORDERS["insert"])
    assert border is not None
    assert _dark_inside(image, border) > 0
    # The lines are 12 points apart, so that the edges of the context fall
    # between two lines.
    _check_context(image, border)


def _edges_dark(shot: tuple[Image.Image, int] | None) -> bool:
    """Tell whether the first or the last row of a zoom has dark pixels."""
    assert shot is not None
    dark = _dark(shot[0])
    return bool(dark[0].any() or dark[-1].any())


def test_no_line_is_cut_at_the_edges_of_a_zoom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zoom grows over the line that its bottom would cut.

    The second paragraph starts 18 points under the first, so that the
    context under the first ends in the second line of the second, which
    has no descender: grown over that line, the zoom has none of its dark
    pixels on its edges.
    """
    path = build_lines_pdf(
        [
            ["First paragraph, its one line."],
            ["One more line here.", "The middle line is here.", "The end."],
        ],
        tmp_path / "gap.pdf",
    )
    first, _ = _load(path).blocks
    request = ShotRequest(
        first.source.boxes,
        "find",
        "insert",
        Wanted("First ", "paragraph", ", its one line."),
    )
    with PdfShooter(path) as shooter:
        assert not _edges_dark(shooter.render(request))
    monkeypatch.setattr(
        "documentation_checker.images.screenshot.EDGE_REACH",
        0.0,
    )
    with PdfShooter(path) as shooter:
        assert _edges_dark(shooter.render(request))


def test_a_whole_block_is_tinted_on_its_page_and_zoomed_whole(
    tmp_path: Path,
) -> None:
    """The page is whole, the block of 41 lines tinted; the zoom holds it."""
    lines = [f"line {i} of a long paragraph that goes on" for i in range(40)]
    path = build_lines_pdf([[*lines, "and ends here."]], tmp_path / "long.pdf")
    (block,) = _load(path).blocks
    request = ShotRequest(block.source.boxes, "whole", "insert")
    page = _page(path, [request])
    with PdfShooter(path) as shooter:
        zoom = shooter.render(request)
    assert page is not None
    assert zoom is not None
    assert page.size == (
        math.ceil(A4[0] * PAGE_SCALE),
        math.ceil(A4[1] * PAGE_SCALE),
    )
    border = _found(page, BORDERS["insert"])
    assert border is not None
    _, top, _, bottom = border
    assert bottom - top == pytest.approx(41 * 12 * PAGE_SCALE, abs=12)
    image, _ = zoom
    border = _found(image, BORDERS["insert"])
    assert border is not None
    _, top, _, bottom = border
    assert bottom - top == pytest.approx(41 * 12 * ZOOM_SCALE, abs=16)
    context = CONTEXT_DOWN * ZOOM_SCALE
    assert image.height == pytest.approx(bottom - top + 2 * context, abs=8)


def test_the_page_shows_the_marks_of_all_its_requests(tmp_path: Path) -> None:
    """The page has the marks of each request on it, each zoom only its own."""
    path = build_lines_pdf(
        [
            ["The quick brown fox jumps over the lazy dog."],
            ["A paragraph that only moved."],
        ],
        tmp_path / "two.pdf",
    )
    first, second = _load(path).blocks
    found = ShotRequest(
        first.source.boxes,
        "find",
        "insert",
        Wanted("The quick ", "brown", " fox"),
    )
    moved = ShotRequest(second.source.boxes, "whole", "block")
    page = _page(path, [found, moved])
    alone = _page(path, [found])
    with PdfShooter(path) as shooter:
        zoom = shooter.render(found)
    assert page is not None
    assert alone is not None
    assert zoom is not None
    assert _found(page, BORDERS["insert"]) is not None
    assert _found(page, BORDERS["block"]) is not None
    assert _found(alone, BORDERS["block"]) is None
    # The zoom of the word shows the next paragraph, but not its tint.
    assert _found(zoom[0], BORDERS["insert"]) is not None
    assert _found(zoom[0], BORDERS["block"]) is None


def test_a_block_that_kept_its_text_is_tinted_in_its_colors(
    tmp_path: Path,
) -> None:
    """A block whose text is kept is tinted whole, and bordered, in blue."""
    path = build_lines_pdf(
        [["A list item that lost its list."]],
        tmp_path / "a.pdf",
    )
    (block,) = _load(path).blocks
    with PdfShooter(path) as shooter:
        shot = shooter.render(
            ShotRequest(block.source.boxes, "whole", "block"),
        )
    assert shot is not None
    border = _found(shot[0], BORDERS["block"])
    assert border is not None
    left, top, right, bottom = border
    pixels = np.asarray(shot[0].convert("RGB"))
    inside = pixels[top + 3 : bottom - 2, left + 3 : right - 2]
    # The light pixels inside are tinted in blue, the text under them dark.
    light = inside[inside.min(axis=-1) > DARK]
    assert len(light)
    assert (light[:, 2] > light[:, 0]).all()
    assert _dark_inside(shot[0], border) > 0


def test_a_part_that_is_not_found_frames_its_blocks_in_its_colors(
    tmp_path: Path,
) -> None:
    """The blocks of a part not found are framed in the colors of the change.

    The frame passes around the block, which is not tinted.
    """
    path = build_lines_pdf([["Some words of the block."]], tmp_path / "a.pdf")
    (block,) = _load(path).blocks
    request = ShotRequest(
        block.source.boxes,
        "find",
        "delete",
        Wanted("", "words that are not there", ""),
    )
    with PdfShooter(path) as shooter:
        shot = shooter.render(request)
    assert shot is not None
    border = _found(shot[0], BORDERS["delete"])
    assert border is not None
    left, top, right, bottom = border
    pixels = np.asarray(shot[0].convert("RGB"))
    inside = pixels[top + 3 : bottom - 2, left + 3 : right - 2]
    # Black text on white, untinted: every pixel inside is a gray.
    assert (inside == inside[..., :1]).all()
    assert _dark_inside(shot[0], border) > 0


def test_marks_that_overlap_are_tinted_once(tmp_path: Path) -> None:
    """Two requests that mark one word give the page of one request."""
    path = build_lines_pdf(
        [["The quick brown fox jumps over the lazy dog."]],
        tmp_path / "lines.pdf",
    )
    (block,) = _load(path).blocks
    request = ShotRequest(
        block.source.boxes,
        "find",
        "mark",
        Wanted("The quick ", "brown", " fox"),
    )
    with PdfShooter(path) as shooter:
        (once,) = shooter.shoot_all([request]).pages
        twice = shooter.shoot_all([request, request]).pages
    assert once is not None
    assert twice == (once, once)


def test_a_wide_page_is_rendered_smaller(tmp_path: Path) -> None:
    """No side of a page or of a zoom has more than `RENDER_SIDE_MAX`."""
    content = "BT /F1 200 Tf 40 100 Td (A very wide line of text) Tj ET"
    path = build_single_page_pdf(
        tmp_path / "wide.pdf",
        content,
        page="/MediaBox [0 0 4000 400]",
    )
    box = PageBox(0, 40.0, 50.0, 3000.0, 300.0)
    request = ShotRequest((box,), "whole", "block")
    page = _page(path, [request])
    with PdfShooter(path) as shooter:
        zoom = shooter.render(request)
    assert zoom is not None
    for image in (page, zoom[0]):
        assert image is not None
        assert max(image.size) <= RENDER_SIDE_MAX
        assert image.width >= RENDER_SIDE_MAX - 2


def test_a_refused_image_stops_the_screenshots_that_show_it(
    tmp_path: Path,
) -> None:
    """The checks of the images of a page come before its renders.

    A refused image leaves its page without screenshot, and the zooms over
    it, but not the zoom of the line of text above it.
    """
    # The image announces more pixels than its stream holds.
    refused = build_image_stream_pdf(
        tmp_path / "refused.pdf",
        b"\x00" * 30,
        filters="",
    )
    valid = build_image_stream_pdf(
        tmp_path / "valid.pdf",
        b"\x80" * (400 * 250 * 3),
        filters="",
    )
    line = ShotRequest((PageBox(0, 72.0, 765.0, 300.0, 780.0),), "whole")
    image = ShotRequest((PageBox(0, 72.0, 600.0, 272.0, 725.0),), "whole")
    with PdfShooter(refused) as shooter:
        assert shooter.render(line) is not None
        assert shooter.render(image) is None
    assert _page(refused, [line]) is None
    with PdfShooter(valid) as shooter:
        assert shooter.render(image) is not None
    assert _page(valid, [line]) is not None


def test_a_missing_file_has_no_screenshot(tmp_path: Path) -> None:
    """A PDF removed since it was read gives no screenshot, nor any error."""
    box = PageBox(0, 72.0, 600.0, 300.0, 700.0)
    request = ShotRequest((box,), "whole", "mark")
    with PdfShooter(tmp_path / "gone.pdf") as shooter:
        assert shooter.render(request) is None
    with PdfShooter(tmp_path / "gone.pdf") as shooter:
        shots = shooter.shoot_all([request, request])
    assert shots == Shots((None, None), (None, None))


@pytest.mark.parametrize(
    "box",
    [
        PageBox(0, math.nan, 600.0, 300.0, 700.0),
        PageBox(0, 72.0, 600.0, math.inf, 700.0),
        PageBox(0, 300.0, 600.0, 72.0, 700.0),
        PageBox(-1, 72.0, 600.0, 300.0, 700.0),
        PageBox(7, 72.0, 600.0, 300.0, 700.0),
    ],
    ids=[
        "nan",
        "infinite",
        "reversed",
        "negative-page",
        "no-page",
    ],
)
def test_boxes_that_cannot_be_shown_have_no_screenshot(
    tmp_path: Path,
    box: PageBox,
) -> None:
    """Boxes not finite, not in order or on no page give no screenshot."""
    path = build_lines_pdf([["Some text."]], tmp_path / "lines.pdf")
    request = ShotRequest((box,), "whole", "mark")
    with PdfShooter(path) as shooter:
        assert shooter.render(request) is None
        assert shooter.shoot_all([request]) == Shots((None,), (None,))


def test_a_closed_shooter_takes_no_screenshot(tmp_path: Path) -> None:
    """Once closed, a shooter no longer opens its PDF."""
    path = build_lines_pdf([["Some text."]], tmp_path / "lines.pdf")
    (block,) = _load(path).blocks
    request = ShotRequest(block.source.boxes, "whole", "mark")
    shooter = PdfShooter(path)
    assert shooter.render(request) is not None
    assert shooter.shoot_all([request]).pages[0] is not None
    shooter.close()
    assert shooter.render(request) is None
    assert shooter.shoot_all([request]) == Shots((None,), (None,))


def test_screenshots_are_the_same_bytes_each_time(tmp_path: Path) -> None:
    """Two shooters give the same zoom and page, byte for byte."""
    path = build_lines_pdf(
        [["The quick brown fox jumps over the lazy dog."]],
        tmp_path / "lines.pdf",
    )
    (block,) = _load(path).blocks
    request = ShotRequest(
        block.source.boxes,
        "find",
        "insert",
        Wanted("The quick ", "brown", " fox"),
    )
    with PdfShooter(path) as first, PdfShooter(path) as second:
        one, two = (
            shooter.shoot_all([request]) for shooter in (first, second)
        )
    assert one == two
    (zoom,), (page,) = one.zooms, one.pages
    assert zoom is not None
    assert page is not None
    assert zoom != page
    assert (zoom.page, zoom.image.data[:4], zoom.image.data[8:12]) == (
        1,
        b"RIFF",
        b"WEBP",
    )


def test_shoot_all_gives_the_screenshots_in_order(tmp_path: Path) -> None:
    """The screenshots made in threads are those made one by one.

    The requests come in any order, and one of them is found on the second
    page of its block: each page shows the marks of the requests on it, and
    they share it.
    """
    lines = [f"line {i} of the long paragraph" for i in range(70)]
    path = build_lines_pdf(
        [["First paragraph here."], lines, ["Last paragraph there."]],
        tmp_path / "two.pdf",
    )
    first, long, last = _load(path).blocks
    requests = [
        ShotRequest(last.source.boxes, "whole", "insert"),
        ShotRequest(
            long.source.boxes,
            "find",
            "delete",
            Wanted("line 65 of the ", "long", " paragraph"),
        ),
        ShotRequest(first.source.boxes, "whole", "insert"),
        ShotRequest((PageBox(0, math.nan, 0.0, 1.0, 1.0),), "whole", "mark"),
    ]
    with PdfShooter(path) as shooter:
        zooms = [encode_shot(shooter.render(request)) for request in requests]
        (one,) = shooter.shoot_all(requests[2:3]).pages
        two, _ = shooter.shoot_all(requests[:2]).pages
    waiting = SHOTS_PER_WORKER * (os.process_cpu_count() or 1)
    rounds = waiting // len(requests) + 2
    with PdfShooter(path) as shooter:
        together = shooter.shoot_all(requests * rounds)
    assert list(together.zooms) == zooms * rounds
    assert list(together.pages) == [two, two, one, None] * rounds
    assert together.pages[0] is together.pages[1]
    assert one is not None
    assert two is not None
    assert (one.page, two.page) == (1, 2)
    assert all(zoom is not None for zoom in zooms[:3])
    assert zooms[-1] is None


def test_shots_decode_to_their_render(tmp_path: Path) -> None:
    """The WebP of a zoom keeps every pixel of its render."""
    path = build_lines_pdf([["Some text to show."]], tmp_path / "lines.pdf")
    (block,) = _load(path).blocks
    request = ShotRequest(block.source.boxes, "whole", "mark")
    with PdfShooter(path) as shooter:
        shot = shooter.render(request)
        (zoom,) = shooter.shoot_all([request]).zooms
    assert shot is not None
    assert zoom is not None
    assert np.array_equal(
        np.asarray(open_raster(zoom.image.data).convert("RGB")),
        np.asarray(shot[0].convert("RGB")),
    )
