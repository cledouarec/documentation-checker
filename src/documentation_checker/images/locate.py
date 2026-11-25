# Copyright 2026 Christophe Le Douarec
"""Find where the part of a text that a finding points at lies on a page.

A side of a finding shows a part of a text, changed or marked, with some text
around it. The screenshots of the reports mark where that part lies on its
page: the characters of the areas of its blocks are read again from the PDF,
and the text is looked for by its letters and digits only, as `match_key`
keeps them, so that the blanks, the punctuation, the accents, the hyphens
and the ligatures, which the extraction and the comparison normalize, do not
stop it from being found. The search is best effort: a text that is not found
gives no place.
"""

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final

from documentation_checker.text.normalize import match_key

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from documentation_checker.images.objects import Box

# The fewest letters and digits that a part of the text needs to be looked for
# without the rest of the text: a shorter one would be found in many places.
SHORTEST_PART: Final = 3
# The share of the height of a character that has to overlap a line for the
# character to be on that line.
LINE_OVERLAP: Final = 0.5


@dataclass(frozen=True, slots=True)
class PageChar:
    """A character of a page.

    Attributes:
        text: The character, as PDFium reads it.
        box: Its left, bottom, right and top edges, from the lower left corner
            of the crop box of the page turned upright.
    """

    text: str
    box: Box


@dataclass(frozen=True, slots=True)
class Area:
    """The characters of a part of a block, on one page.

    Attributes:
        page: Position of the page, from 0.
        chars: The characters of the page whose centre lies in the box of the
            part, in the order of PDFium.
    """

    page: int
    chars: tuple[PageChar, ...]


@dataclass(frozen=True, slots=True)
class Wanted:
    """The text of a side of a finding, cut around the part it points at.

    Attributes:
        before: The text before the part.
        changed: The part, which is empty where text was only removed.
        after: The text after the part.
    """

    before: str
    changed: str
    after: str


@dataclass(frozen=True, slots=True)
class Spot:
    """Where the part of a text lies on a page.

    Attributes:
        page: Position of the page, from 0.
        marks: The box of the part on each of its lines, in reading order;
            for a caret, one box without width at the place of the part.
        caret: Whether the part is only a place, where text was removed.
    """

    page: int
    marks: tuple[Box, ...]
    caret: bool = False


@dataclass(frozen=True, slots=True)
class _Haystack:
    """The keys of the characters of some areas, joined.

    Attributes:
        text: The key of each character, as `match_key` gives it, joined.
        origins: For each character of ``text``, the area and the position
            in it of the character whose key holds it.
    """

    text: str
    origins: tuple[tuple[int, int], ...]


@cache
def _char_key(text: str) -> str:
    """Return the key of a character, as `match_key` gives it."""
    return match_key(text)


def _haystack(areas: Sequence[Area]) -> _Haystack:
    """Return the keys of the characters of some areas, joined in order."""
    keys: list[str] = []
    origins: list[tuple[int, int]] = []
    for area_index, area in enumerate(areas):
        for char_index, char in enumerate(area.chars):
            key = _char_key(char.text)
            keys.append(key)
            origins.extend([(area_index, char_index)] * len(key))
    return _Haystack("".join(keys), tuple(origins))


def _found(text: str, needle: str, *, unique: bool) -> int | None:
    """Return where a needle is in a text: first, or only, if it must be."""
    first = text.find(needle)
    if first < 0 or (unique and text.find(needle, first + 1) >= 0):
        return None
    return first


def _needles(
    before: str,
    changed: str,
    after: str,
) -> Iterator[tuple[str, int, bool]]:
    """Yield what to look for, in order, until the part is found.

    Each needle comes with the place of the part in it and whether it has to
    be found once only. The whole text comes first, at its first place, if it
    is long enough to be told apart; then the part with the text before it,
    with the text after it, and alone, each of them once only, and long
    enough to be told apart.
    """
    whole = before + changed + after
    if whole:
        yield whole, len(before), len(whole) < SHORTEST_PART
    for needle, offset in (
        (before + changed, len(before)),
        (changed + after, 0),
        (changed, 0),
    ):
        if needle != whole and len(needle) >= SHORTEST_PART:
            yield needle, offset, True


def _span(hay: str, wanted: Wanted) -> tuple[int, int] | None:
    """Return the start and the end of the part among the keys of a page.

    Args:
        hay: The keys of the characters, joined.
        wanted: The text to find.

    Returns:
        The start and the end of the keys of the part; the same place twice
        for a part without letters or digits, between the text before it and
        the text after it. ``None`` if the part is not found.
    """
    before = match_key(wanted.before)
    changed = match_key(wanted.changed)
    after = match_key(wanted.after)
    for needle, offset, unique in _needles(before, changed, after):
        place = _found(hay, needle, unique=unique)
        if place is not None:
            start = place + offset
            return start, start + len(changed)
    return None


def _between(
    hay: _Haystack,
    start: int,
    end: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Return the first and the last character of a part, by their origins.

    Args:
        hay: The keys of the characters of the areas.
        start: The start of the keys of the part.
        end: The end of the keys of the part, ``start`` for a part without
            letters or digits.

    Returns:
        The origins of the first and of the last character of the part, both
        included: for a part without letters or digits, those strictly
        between the text before it and the text after it, which may hold no
        character. ``None`` if the part is out of the keys.
    """
    origins = hay.origins
    if end > start:
        return origins[start], origins[end - 1]
    if not 0 < start < len(origins):
        return None
    last_before, first_after = origins[start - 1], origins[start]
    return (
        (last_before[0], last_before[1] + 1),
        (first_after[0], first_after[1] - 1),
    )


def _chars(
    areas: Sequence[Area],
    first: tuple[int, int],
    last: tuple[int, int],
) -> list[tuple[int, PageChar]]:
    """Return the characters from one origin to another, with their pages."""
    chars: list[tuple[int, PageChar]] = []
    for area_index in range(first[0], last[0] + 1):
        area = areas[area_index]
        start = first[1] if area_index == first[0] else 0
        stop = last[1] + 1 if area_index == last[0] else len(area.chars)
        chars.extend((area.page, char) for char in area.chars[start:stop])
    return chars


def _same_line(line: Box, box: Box) -> bool:
    """Return whether a character is on a line, by their heights."""
    overlap = min(line[3], box[3]) - max(line[1], box[1])
    return overlap >= LINE_OVERLAP * (box[3] - box[1])


def _lines(boxes: Sequence[Box]) -> tuple[Box, ...]:
    """Return the boxes of the lines of some characters, in their order."""
    lines: list[Box] = []
    for box in boxes:
        if lines and _same_line(lines[-1], box):
            left, bottom, right, top = lines[-1]
            lines[-1] = (
                min(left, box[0]),
                min(bottom, box[1]),
                max(right, box[2]),
                max(top, box[3]),
            )
        else:
            lines.append(box)
    return tuple(lines)


def _caret(
    areas: Sequence[Area],
    hay: _Haystack,
    start: int,
) -> Spot:
    """Return the place between the text before a part and the text after.

    Args:
        areas: The characters of the blocks.
        hay: Their keys, at least one.
        start: The place of the part among the keys.

    Returns:
        A caret in the middle of the gap between the last character before
        the place and the first one after it, when both are on one line;
        else at the right edge of the one before, or at the left edge of the
        one after for the very first place.
    """
    origins = hay.origins
    if not start:
        page, char = _char(areas, origins[0])
        x = char.box[0]
    else:
        page, char = _char(areas, origins[start - 1])
        x = char.box[2]
        if start < len(origins):
            after_page, after = _char(areas, origins[start])
            if after_page == page and _same_line(char.box, after.box):
                x = (x + after.box[0]) / 2
    _, bottom, _, top = char.box
    return Spot(page, ((x, bottom, x, top),), caret=True)


def _char(
    areas: Sequence[Area],
    origin: tuple[int, int],
) -> tuple[int, PageChar]:
    """Return the page of a character and the character, by its origin."""
    area = areas[origin[0]]
    return area.page, area.chars[origin[1]]


def locate(areas: Sequence[Area], wanted: Wanted) -> Spot | None:
    """Find the part of a text among the characters of the areas of its blocks.

    The keys of the characters of the areas, joined in order, are searched
    for the whole text, then for the part with the text before it, with the
    text after it, then alone, as `_needles` tells. The characters of the
    part are those from its first letter or digit to its last; for a part
    without any, those between the text before it and the text after it, or
    else the place between them, as a caret. Blank characters are left out,
    and so are the characters of the part on other pages than its first.

    Args:
        areas: The characters of the blocks of the side, in reading order.
        wanted: The text of the side, cut around its part.

    Returns:
        Where the part lies, ``None`` if it is not found.
    """
    hay = _haystack(areas)
    span = _span(hay.text, wanted)
    if span is None:
        return None
    start, end = span
    bounds = _between(hay, start, end)
    chars = [] if bounds is None else _chars(areas, *bounds)
    boxes = [
        char.box
        for page, char in chars
        if page == chars[0][0] and not char.text.isspace()
    ]
    if boxes:
        return Spot(chars[0][0], _lines(boxes))
    if start == end:
        return _caret(areas, hay, start)
    return None
