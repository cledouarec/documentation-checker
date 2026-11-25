# Copyright 2026 Christophe Le Douarec
"""Tests for finding the part of a text that a finding points at on a page."""

from typing import TYPE_CHECKING

from documentation_checker.images.locate import (
    Area,
    PageChar,
    Spot,
    Wanted,
    locate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# Each character of `_chars` is 5 points wide and 10 high, a line every 12
# points, from 72 points on the left and 700 points up.
LEFT = 72.0
WIDTH = 5.0
BOTTOM = 700.0
HEIGHT = 10.0
LEADING = 12.0


def _chars(lines: Sequence[str]) -> tuple[PageChar, ...]:
    """Make the characters of lines of text, one box after the other."""
    chars = []
    for row, line in enumerate(lines):
        bottom = BOTTOM - row * LEADING
        for column, text in enumerate(line):
            left = LEFT + column * WIDTH
            chars.append(
                PageChar(text, (left, bottom, left + WIDTH, bottom + HEIGHT)),
            )
    return tuple(chars)


def _box(start: int, end: int, row: int = 0) -> tuple[float, ...]:
    """Return the box of the characters from one column to another, both in."""
    bottom = BOTTOM - row * LEADING
    return (
        LEFT + start * WIDTH,
        bottom,
        LEFT + (end + 1) * WIDTH,
        bottom + HEIGHT,
    )


def _locate(lines: Sequence[str], wanted: Wanted) -> Spot | None:
    """Find a part of a text in lines of one page."""
    return locate([Area(0, _chars(lines))], wanted)


def test_a_changed_word_is_found_with_its_context() -> None:
    """The part is marked where the text around it is found."""
    lines = ["The quick brown fox jumps over the lazy dog."]
    spot = _locate(lines, Wanted("The quick ", "brown", " fox jumps"))
    assert spot == Spot(0, (_box(10, 14),))


def test_a_part_on_two_lines_has_a_mark_on_each() -> None:
    """A part that a line break cuts is marked on both of its lines."""
    lines = ["one two three", "four five six"]
    spot = _locate(lines, Wanted("one two ", "three four", " five"))
    assert spot == Spot(0, (_box(8, 12), _box(0, 3, row=1)))


def test_a_change_of_punctuation_marks_the_punctuation() -> None:
    """A part without letters or digits marks what lies between its texts."""
    spot = _locate(["first, second"], Wanted("first", ",", " second"))
    assert spot == Spot(0, (_box(5, 5),))


def test_removed_text_leaves_a_caret_in_the_gap() -> None:
    """A place where text was removed is a caret between two words."""
    spot = _locate(["over the dog"], Wanted("over the", "", " dog"))
    middle = (LEFT + 8 * WIDTH + LEFT + 9 * WIDTH) / 2
    assert spot == Spot(
        0,
        ((middle, BOTTOM, middle, BOTTOM + HEIGHT),),
        caret=True,
    )


def test_removed_text_at_the_end_leaves_a_caret_after_the_text() -> None:
    """A place after the last character is a caret at its right edge."""
    spot = _locate(["the end"], Wanted("the end", "", ""))
    right = LEFT + 7 * WIDTH
    assert spot == Spot(
        0,
        ((right, BOTTOM, right, BOTTOM + HEIGHT),),
        caret=True,
    )


def test_ligatures_hyphens_and_quotes_do_not_stop_the_search() -> None:
    """The characters that the comparison normalizes are not compared."""
    lines = ["the ﬁrst connec-", "tion is “fast”"]
    wanted = Wanted("the first ", "connection", ' is "fast"')
    spot = _locate(lines, wanted)
    assert spot == Spot(0, (_box(9, 15), _box(0, 3, row=1)))


def test_accents_and_case_do_not_stop_the_search() -> None:
    """A text is found whatever its accents and its case."""
    spot = _locate(["Un été très chaud"], Wanted("un ", "ETE", ""))
    assert spot == Spot(0, (_box(3, 5),))


def test_a_repeated_word_alone_is_not_found() -> None:
    """A part found in two places, without the text around it, has no place."""
    lines = ["the cat and the dog"]
    assert _locate(lines, Wanted("zzz ", "the", " yyy")) is None


def test_a_part_found_once_is_found_without_its_text() -> None:
    """A part found once only is found though the text around it is not."""
    lines = ["the cat and the dog"]
    spot = _locate(lines, Wanted("zzz ", "cat", " yyy"))
    assert spot == Spot(0, (_box(4, 6),))


def test_a_short_part_alone_is_not_found() -> None:
    """A part of fewer than three letters or digits needs its text."""
    assert _locate(["a b c"], Wanted("zzz ", "b", " yyy")) is None


def test_a_text_across_a_page_break_is_found() -> None:
    """The areas of a block on two pages are searched as one text."""
    areas = [
        Area(3, _chars(["end of the first page"])),
        Area(4, _chars(["start of the next one"])),
    ]
    spot = locate(areas, Wanted("first page ", "start", " of the"))
    assert spot == Spot(4, (_box(0, 4),))


def test_a_part_across_a_page_break_keeps_its_first_page() -> None:
    """The marks of a part on two pages are those of its first page."""
    areas = [
        Area(3, _chars(["end of the first page"])),
        Area(4, _chars(["start of the next one"])),
    ]
    spot = locate(areas, Wanted("the ", "first page start", " of"))
    assert spot == Spot(3, (_box(11, 20),))


def test_a_text_that_is_not_there_is_not_found() -> None:
    """A text that the page does not hold has no place."""
    assert _locate(["some text"], Wanted("other ", "words", "")) is None


def test_an_empty_text_is_not_found() -> None:
    """A side without letters or digits has no place."""
    assert _locate(["some text"], Wanted("", "", "")) is None
    assert _locate(["some text"], Wanted("...", "", "")) is None
