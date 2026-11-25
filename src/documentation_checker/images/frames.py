# Copyright 2026 Christophe Le Douarec
"""The two frames of a PDF page: as PDFium places it, and turned upright.

PDFium gives the boxes of a page in its own coordinates, those of the page
before any rotation. The reader turns each page upright, by quarter turns,
so that its text runs from left to right, and measures from the lower left
corner of its crop box turned so. These helpers go from one frame to the
other, and tell the rotation that makes PDFium render a page upright.
"""

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from documentation_checker.images.objects import Box

QUARTER: Final = 90  # Degrees in a quarter turn.
QUARTERS: Final = 4  # Quarter turns in a full turn.


def _turned(box: Box, turn: int, size: tuple[float, float]) -> Box:
    """Return where a box is once its page is turned upright.

    Args:
        box: The left, bottom, right and top edges of the box, from the lower
            left corner of the crop box.
        turn: The quarter turns, clockwise, of the text of the page.
        size: The width and the height of the crop box.

    Returns:
        The left, bottom, right and top edges of the box, from the lower left
        corner of the crop box turned so that the text of the page runs from
        left to right.
    """
    left, bottom, right, top = box
    width, height = size
    match turn:
        case 0:
            return left, bottom, right, top
        case 1:
            return height - top, left, height - bottom, right
        case 2:
            return width - right, height - top, width - left, height - bottom
        case _:
            return bottom, width - right, top, width - left


def upright_size(crop: Box, turn: int) -> tuple[float, float]:
    """Return the width and the height of a crop box turned upright.

    Args:
        crop: The left, bottom, right and top edges of the crop box, in the
            coordinates of PDFium for the page.
        turn: The quarter turns, clockwise, of the text of the page.

    Returns:
        The width and the height, which an odd number of turns swaps.
    """
    left, bottom, right, top = crop
    width, height = right - left, top - bottom
    return (height, width) if turn % 2 else (width, height)


def to_upright(box: Box, crop: Box, turn: int) -> Box:
    """Return a box of PDFium in the frame of the page turned upright.

    Args:
        box: The left, bottom, right and top edges of the box, in the
            coordinates of PDFium for the page.
        crop: The crop box of the page, in the same coordinates.
        turn: The quarter turns, clockwise, of the text of the page.

    Returns:
        The left, bottom, right and top edges of the box, from the lower left
        corner of the crop box turned upright.
    """
    left, bottom, right, top = box
    crop_left, crop_bottom, crop_right, crop_top = crop
    return _turned(
        (
            left - crop_left,
            bottom - crop_bottom,
            right - crop_left,
            top - crop_bottom,
        ),
        turn,
        (crop_right - crop_left, crop_top - crop_bottom),
    )


def from_upright(box: Box, crop: Box, turn: int) -> Box:
    """Return a box of the page turned upright in the coordinates of PDFium.

    The reverse of `to_upright`: the page turned upright is turned back by
    the quarter turns that complete a full turn.

    Args:
        box: The left, bottom, right and top edges of the box, from the lower
            left corner of the crop box turned upright.
        crop: The crop box of the page, in the coordinates of PDFium.
        turn: The quarter turns, clockwise, of the text of the page.

    Returns:
        The left, bottom, right and top edges of the box, in the coordinates
        of PDFium for the page.
    """
    back = (QUARTERS - turn) % QUARTERS
    x0, y0, x1, y1 = _turned(box, back, upright_size(crop, turn))
    crop_left, crop_bottom = crop[0], crop[1]
    return x0 + crop_left, y0 + crop_bottom, x1 + crop_left, y1 + crop_bottom


def upright_rotation(turn: int, rotation: int) -> int:
    """Return the rotation that makes PDFium render a page upright.

    A page whose text the reader turned by ``turn`` quarter turns is shown
    upright by a clockwise rotation of the quarter turns that complete a
    full turn. PDFium already applies the rotation that the page declares.

    Args:
        turn: The quarter turns, clockwise, of the text of the page.
        rotation: The rotation that the page declares, in degrees, as
            ``get_rotation`` of pypdfium2 reads its ``/Rotate``.

    Returns:
        The rotation to add when the page is rendered, in degrees: 0, 90,
        180 or 270.
    """
    upright = (QUARTERS - turn) % QUARTERS * QUARTER
    return (upright - rotation) % (QUARTERS * QUARTER)
