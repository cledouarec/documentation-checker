# Copyright 2026 Christophe Le Douarec
"""Screenshots of the pages of PDF documents that the findings point at.

Each side of a finding gets a zoom: the area of its page around what it
points at, the lines that it marks across their blocks with about two lines
of the page above and under them, no line cut, rendered at `ZOOM_SCALE`.
Each page that the findings point at gets a screenshot of the whole page, at
`PAGE_SCALE`, with the marks of all of them, so that their changes are seen
in their context. Both are turned upright, as the reader reads the page.

A screenshot marks what a finding points at: the part of the text that it
changed or marked, tinted and bordered, where `locate` finds it, or a caret
where text was only removed; the whole blocks, tinted, for a block missing
or added, or whose text is kept but not its place or its kind; and a frame
around the blocks, in the colors of their change, for a part that is not
found. The page is rendered by PDFium after the checks of `RenderGuard`, and
the screenshot is encoded again in WebP, so that no byte of the document
comes through.
"""

import math
import os
from array import array
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Self

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image, ImageChops, ImageDraw

from documentation_checker.images.frames import (
    from_upright,
    to_upright,
    upright_size,
)
from documentation_checker.images.load import (
    RenderGuard,
    fitted_scale,
    render_upright,
)
from documentation_checker.images.locate import (
    Area,
    PageChar,
    Spot,
    Wanted,
    locate,
)
from documentation_checker.images.objects import boxes_meet
from documentation_checker.images.pixels import UnreadableImage
from documentation_checker.images.thumbnail import encode_webp
from documentation_checker.text.codes import (
    LINE_BREAKS,
    char_text,
    join_surrogates,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from concurrent.futures import Future
    from pathlib import Path
    from types import TracebackType

    from documentation_checker.images.objects import Box
    from documentation_checker.images.thumbnail import Thumbnail
    from documentation_checker.model.document import PageBox

PAGE_SCALE: Final = 1.5  # Pixels for each point of a page: 108 DPI.
ZOOM_SCALE: Final = 2.0  # Pixels for each point of a zoom: 144 DPI.
# The points that a zoom shows around what it marks: on the left and on the
# right of the blocks, and above and under the lines, about two of them.
CONTEXT_ACROSS: Final = 12.0
CONTEXT_DOWN: Final = 24.0
ZOOM_MIN_WIDTH: Final = 300.0  # Fewest points across a zoom.
# The points rendered above and under a zoom, where its top and its bottom
# may move so as not to cut a line: about one line.
EDGE_REACH: Final = 12.0
INK: Final = 160  # The luma under which a pixel is ink, as the text is.
# The lossless WebP encoding of a screenshot, its effort and its method: it
# keeps the text sharp in fewer bytes than a lossy encoding, and as fast.
# libwebp searches further past the efforts 25, 50 and 75: 51 gives a page in
# 14% fewer bytes than 26 to 50, for 6% more time.
SHOT_EFFORT: Final = 51
SHOT_METHOD: Final = 1
# The screenshots waiting for their marks and their encoding, for each thread
# that makes them: enough to keep the threads busy.
SHOTS_PER_WORKER: Final = 2
# The pages kept open, with their checks and their characters: a block over
# two pages needs both, and the next screenshot is mostly on one of them.
OPEN_PAGES: Final = 2
LINE_WIDTH: Final = 2  # Pixels of a border, of a frame and of a caret.
MARK_PADDING: Final = 1  # Pixels added around each mark.
# How far out of the box of an area the centre of a character may lie, in
# points: the boxes of the areas are measured from the font sizes.
AREA_TOLERANCE: Final = 2.0

# What a screenshot marks: the part of the text that the finding points at,
# or the blocks whole.
type ShotMode = Literal["find", "whole"]
# The colors of the marks: those of a text deleted, inserted or marked, and
# those of a block whose text is kept, but not its place or its kind.
type ShotStyle = Literal["delete", "insert", "mark", "block"]

# The tints of the marks, multiplied into the render so that the text under
# them stays dark, and the colors of their borders and carets.
TINTS: Final[Mapping[ShotStyle, tuple[int, int, int]]] = MappingProxyType(
    {
        "delete": (255, 190, 186),
        "insert": (176, 240, 190),
        "mark": (255, 226, 130),
        "block": (182, 212, 255),
    },
)
BORDERS: Final[Mapping[ShotStyle, tuple[int, int, int]]] = MappingProxyType(
    {
        "delete": (207, 34, 46),
        "insert": (26, 127, 55),
        "mark": (154, 103, 0),
        "block": (9, 105, 218),
    },
)


@dataclass(frozen=True, slots=True)
class Screenshot:
    """A screenshot of a page, or of an area of a page, encoded.

    Attributes:
        image: The screenshot, in WebP.
        page: Number of its page, from 1.
    """

    image: Thumbnail
    page: int


@dataclass(frozen=True, slots=True)
class ShotRequest:
    """What the screenshot of a side of a finding shows.

    Attributes:
        boxes: The boxes of the blocks of the side, as their source gives
            them, at least one.
        mode: ``"find"`` to mark the part of the text that ``wanted`` points
            at, or to frame the blocks if it is not found, ``"whole"`` to
            tint the blocks whole.
        style: The colors of the marks.
        wanted: The text of the side, cut around its part, for ``"find"``.
    """

    boxes: tuple[PageBox, ...]
    mode: ShotMode
    style: ShotStyle = "mark"
    wanted: Wanted | None = None


@dataclass(frozen=True, slots=True)
class Shots:
    """The screenshots of some requests.

    Attributes:
        zooms: The zoom of each request, in their order; ``None`` for a
            request without one.
        pages: The page that each request marks, in their order, with the
            marks of all the requests on it, which share it; ``None`` for a
            request that cannot be shown, or whose page cannot be rendered.
    """

    zooms: tuple[Screenshot | None, ...]
    pages: tuple[Screenshot | None, ...]


class _PageText:
    """The characters of a page, read from its text page as they are needed.

    The loose box of each character, as high as the line of its font, is read
    once, in the coordinates of PDFium; the text of a character only when an
    area holds it, as `char_text` tells it, the surrogate halves joined by
    `join_surrogates`. The line breaks that PDFium generates are left out.
    """

    def __init__(self, page: pdfium.PdfPage, crop: Box, turn: int) -> None:
        """Open the text page of a page and read the boxes of its characters.

        Args:
            page: The page.
            crop: Its crop box, in the coordinates of PDFium.
            turn: The quarter turns, clockwise, of the text of the page.
        """
        self._textpage = page.get_textpage()
        self._crop = crop
        self._turn = turn
        self._held: dict[Box, tuple[PageChar, ...]] = {}
        raw = self._textpage.raw
        loose_box = pdfium_c.FPDFText_GetLooseCharBox
        rect = pdfium_c.FS_RECTF()
        indices: list[int] = []
        edges = array("d")
        for index in range(self._textpage.count_chars()):
            if loose_box(raw, index, rect):
                indices.append(index)
                edges.extend((rect.left, rect.bottom, rect.right, rect.top))
        self._indices = indices
        self._boxes = np.frombuffer(edges, dtype=float).reshape(-1, 4)
        self._x = (self._boxes[:, 0] + self._boxes[:, 2]) / 2
        self._y = (self._boxes[:, 1] + self._boxes[:, 3]) / 2

    def close(self) -> None:
        """Close the text page."""
        self._textpage.close()

    def inside(self, area: Box) -> tuple[PageChar, ...]:
        """Return the characters whose centre lies in an area of the page.

        Args:
            area: The left, bottom, right and top edges of the area, from the
                lower left corner of the crop box turned upright.

        Returns:
            The characters whose centre lies at most `AREA_TOLERANCE` points
            out of the area, in the order of PDFium, their boxes from the
            lower left corner of the crop box turned upright; the same ones
            each time for one area.
        """
        if area not in self._held:
            self._held[area] = self._read(area)
        return self._held[area]

    def _read(self, area: Box) -> tuple[PageChar, ...]:
        """Return the characters of an area, read from the text page."""
        left, bottom, right, top = from_upright(area, self._crop, self._turn)
        held = np.flatnonzero(
            (self._x >= left - AREA_TOLERANCE)
            & (self._x <= right + AREA_TOLERANCE)
            & (self._y >= bottom - AREA_TOLERANCE)
            & (self._y <= top + AREA_TOLERANCE),
        )
        raw = self._textpage.raw
        chars: list[PageChar] = []
        for position, box in zip(
            held.tolist(),
            self._boxes[held].tolist(),
            strict=True,
        ):
            index = self._indices[position]
            code = pdfium_c.FPDFText_GetUnicode(raw, index)
            if code in LINE_BREAKS:
                continue
            unmapped = pdfium_c.FPDFText_HasUnicodeMapError(raw, index) == 1
            text = char_text(code, hyphen=False, unmapped=unmapped)
            upright = to_upright(tuple(box), self._crop, self._turn)
            chars.append(PageChar(text, upright))
        return tuple(join_surrogates(chars)[0])


def _holding(blocks: Sequence[Box], marks: Sequence[Box]) -> list[Box]:
    """Return the boxes of the blocks that meet a mark, else all of them."""
    holding = [
        block
        for block in blocks
        if any(boxes_meet(mark, block) for mark in marks)
    ]
    return holding or list(blocks)


def _core(blocks: Sequence[Box], marks: Sequence[Box]) -> Box:
    """Return the box of what a zoom shows, before the context around it.

    Args:
        blocks: The boxes of the blocks on the page, at least one.
        marks: The marks of the part of the text, if it was found.

    Returns:
        The lines of the marks, or the blocks without marks, across the
        blocks that hold the marks.
    """
    across = [*_holding(blocks, marks), *marks]
    down = marks or blocks
    return (
        min(box[0] for box in across),
        min(box[1] for box in down),
        max(box[2] for box in across),
        max(box[3] for box in down),
    )


def zoom_area(
    blocks: Sequence[Box],
    marks: Sequence[Box],
    size: tuple[float, float],
) -> Box | None:
    """Return the area of a page that the zoom of a screenshot shows.

    Args:
        blocks: The boxes of the blocks on the page, at least one.
        marks: The marks of the part of the text, if it was found.
        size: The width and the height of the page turned upright.

    Returns:
        The lines of the marks, or the blocks without marks, across the
        blocks that hold the marks, with `CONTEXT_ACROSS` points on the left
        and on the right and `CONTEXT_DOWN` points above and under, widened
        around its middle to `ZOOM_MIN_WIDTH` points, all within the page.
        A widening that crosses a side of the page is moved back within it,
        as far as the widening goes. ``None`` if nothing of the lines or of
        the blocks, with the points around them, is on the page.
    """
    left, bottom, right, top = _core(blocks, marks)
    left, right = left - CONTEXT_ACROSS, right + CONTEXT_ACROSS
    bottom, top = bottom - CONTEXT_DOWN, top + CONTEXT_DOWN
    width, height = size
    if right <= 0 or left >= width or top <= 0 or bottom >= height:
        return None
    missing = ZOOM_MIN_WIDTH - (right - left)
    if missing > 0:
        left, right = left - missing / 2, right + missing / 2
        shift = max(-left, 0.0) - max(right - width, 0.0)
        shift = min(max(shift, -missing / 2), missing / 2)
        left, right = left + shift, right + shift
    return (
        max(left, 0.0),
        max(bottom, 0.0),
        min(right, width),
        min(top, height),
    )


def quiet_rows(image: Image.Image, columns: tuple[int, int]) -> np.ndarray:
    """Return which rows of a render hold no ink between two columns.

    Args:
        image: The render.
        columns: The first column of pixels, and the one after the last.

    Returns:
        For each row, whether none of its pixels between the columns is
        darker than `INK`; the columns that are ink on every row, such as
        the rules of a table, are left out.
    """
    luma = np.asarray(image.convert("L"))[:, columns[0] : columns[1]]
    ink = luma < INK
    return ~ink[:, ~ink.all(axis=0)].any(axis=1)


def edge_row(
    quiet: Sequence[bool],
    row: int,
    *,
    outward: Literal[-1, 1],
    reach: int,
    limit: int,
) -> int:
    """Return the row of an edge of a zoom, on a row without ink if it can.

    Args:
        quiet: Whether each row of the render holds no ink.
        row: The row where the edge would be.
        outward: ``-1`` for the top edge, ``1`` for the bottom edge: the way
            in which the edge moves to show more.
        reach: Most rows that the edge moves by.
        limit: The row that the edge never moves past inward, the first or
            the last of what the zoom marks.

    Returns:
        The nearest quiet row outward, within the reach and the render, so
        that the edge holds the whole line that it cut; else the nearest
        one inward, within the reach and not past the limit, so that it
        leaves that line out; else the row, within the render.
    """
    last = len(quiet) - 1
    row = min(max(row, 0), last)
    for distance in range(reach + 1):
        candidate = row + outward * distance
        if 0 <= candidate <= last and quiet[candidate]:
            return candidate
    for distance in range(1, reach + 1):
        candidate = row - outward * distance
        if (candidate - limit) * outward < 0 or not 0 <= candidate <= last:
            break
        if quiet[candidate]:
            return candidate
    return row


@dataclass(frozen=True, slots=True)
class _Canvas:
    """The pixels of the render of a page, or of an area of a page.

    Attributes:
        height: The height of the page turned upright, in points.
        scale: Pixels for each point.
        left: The pixels of the page on the left of the render.
        top: The pixels of the page above the render.
    """

    height: float
    scale: float
    left: int
    top: int

    @classmethod
    def of_area(cls, area: Box, height: float, scale: float) -> Self:
        """Return the pixels of the render of an area, as PDFium crops it.

        Args:
            area: The left, bottom, right and top edges of the area, in
                points of the page turned upright.
            height: The height of the page turned upright, in points.
            scale: Pixels for each point.

        Returns:
            The pixels, whose first ones are those that PDFium keeps once it
            cuts the whole pixels of the page left of the area and above it.
        """
        return cls(
            height,
            scale,
            math.ceil(area[0] * scale),
            math.ceil((height - area[3]) * scale),
        )

    def pixels(self, box: Box, *, padding: int = 0) -> Box:
        """Return the left, top, right and bottom pixels of a box."""
        scale = self.scale
        return (
            box[0] * scale - self.left - padding,
            (self.height - box[3]) * scale - self.top - padding,
            box[2] * scale - self.left + padding,
            (self.height - box[1]) * scale - self.top + padding,
        )


def _tinted(
    image: Image.Image,
    canvas: _Canvas,
    boxes: Sequence[Box],
    style: ShotStyle,
) -> Image.Image:
    """Return a render with boxes tinted in a style, in one layer.

    The tint is multiplied into the part of the render that the boxes cover
    only, so that marking a word costs no more than the word, and once, so
    that boxes that overlap are not darker where they do.
    """
    shown = [canvas.pixels(box, padding=MARK_PADDING) for box in boxes]
    left = max(math.floor(min(pixels[0] for pixels in shown)), 0)
    top = max(math.floor(min(pixels[1] for pixels in shown)), 0)
    right = min(math.ceil(max(pixels[2] for pixels in shown)) + 1, image.width)
    bottom = min(
        math.ceil(max(pixels[3] for pixels in shown)) + 1,
        image.height,
    )
    if right > left and bottom > top:
        part = image.crop((left, top, right, bottom))
        layer = Image.new(image.mode, part.size, "white")
        tint = ImageDraw.Draw(layer)
        for x0, y0, x1, y1 in shown:
            tint.rectangle(
                (x0 - left, y0 - top, x1 - left, y1 - top),
                fill=TINTS[style],
            )
        image.paste(ImageChops.multiply(part, layer), (left, top))
    return image


@dataclass(slots=True)
class _Marks:
    """The marks of some screenshots, in points of the page turned upright.

    Attributes:
        tints: The boxes tinted and bordered, by the style of their colors.
        frames: The boxes framed, each with the style of its color.
        carets: The places of carets, each with the style of its color.
    """

    tints: dict[ShotStyle, list[Box]] = field(default_factory=dict)
    frames: list[tuple[Box, ShotStyle]] = field(default_factory=list)
    carets: list[tuple[Box, ShotStyle]] = field(default_factory=list)

    def draw(self, image: Image.Image, canvas: _Canvas) -> Image.Image:
        """Return a render with the marks: the tints, then the lines.

        The borders, the frames and the carets are drawn over all the tints,
        so that none of them is tinted. The styles come in the order of
        `TINTS`, whatever the order of the marks.
        """
        tints = [
            (style, self.tints[style])
            for style in TINTS
            if style in self.tints
        ]
        for style, boxes in tints:
            image = _tinted(image, canvas, boxes, style)
        draw = ImageDraw.Draw(image)
        for style, boxes in tints:
            for box in boxes:
                draw.rectangle(
                    canvas.pixels(box, padding=MARK_PADDING),
                    outline=BORDERS[style],
                    width=LINE_WIDTH,
                )
        for box, style in self.frames:
            draw.rectangle(
                canvas.pixels(box, padding=LINE_WIDTH + MARK_PADDING),
                outline=BORDERS[style],
                width=LINE_WIDTH,
            )
        for place, style in self.carets:
            left, top, _, bottom = canvas.pixels(place, padding=MARK_PADDING)
            x = left + MARK_PADDING
            draw.rectangle(
                (x - LINE_WIDTH / 2, top, x + LINE_WIDTH / 2, bottom),
                fill=BORDERS[style],
            )
        return image


def _valid(boxes: Sequence[PageBox]) -> bool:
    """Return whether boxes can be shown: finite and in order.

    A page that the PDF does not have is refused by PDFium.
    """
    return bool(boxes) and all(
        all(math.isfinite(edge) for edge in box.edges)
        and box.x0 <= box.x1
        and box.bottom <= box.top
        for box in boxes
    )


def _first_page(request: ShotRequest) -> int:
    """Return the first page of the boxes of a request, by its position."""
    return min((box.page for box in request.boxes), default=0)


@dataclass(frozen=True, slots=True)
class _Planned:
    """Where a request shows on its page, and what it marks there.

    Attributes:
        request: What the screenshot shows.
        page: Position of the page, from 0.
        turn: The quarter turns, clockwise, of the text of the page.
        size: The width and the height of the page turned upright.
        blocks: The boxes of the blocks of the request on the page.
        spot: Where the part of the text lies, if it is found.
    """

    request: ShotRequest
    page: int
    turn: int
    size: tuple[float, float]
    blocks: tuple[Box, ...]
    spot: Spot | None

    def mark(self, marks: _Marks) -> None:
        """Add the marks of the request to some marks.

        Args:
            marks: The marks of a screenshot, which this one adds to.
        """
        request, spot = self.request, self.spot
        if request.mode == "whole":
            marks.tints.setdefault(request.style, []).extend(self.blocks)
        elif spot is None:
            marks.frames.extend((box, request.style) for box in self.blocks)
        elif spot.caret:
            marks.carets.append((spot.marks[0], request.style))
        else:
            marks.tints.setdefault(request.style, []).extend(spot.marks)


@dataclass(frozen=True, slots=True)
class _Trim:
    """Where a zoom ends, once its render is cut.

    Attributes:
        area: The area of the zoom, as `zoom_area` gives it, which its render
            passes by `EDGE_REACH` points above and under.
        core: The box of what it marks, which its edges never cut into, and
            whose columns tell where the rows have ink.
    """

    area: Box
    core: Box


def _trimmed(
    image: Image.Image,
    canvas: _Canvas,
    trim: _Trim,
) -> tuple[Image.Image, _Canvas]:
    """Return the render of a zoom cut to its area, and its pixels.

    The top and the bottom of the area move to rows without ink, as
    `edge_row` tells, among the columns of what the zoom marks, so that no
    line is cut there.
    """
    _, top, _, bottom = canvas.pixels(trim.area)
    core = canvas.pixels(trim.core)
    columns = (
        max(math.floor(core[0]), 0),
        min(math.ceil(core[2]), image.width),
    )
    quiet = quiet_rows(image, columns)
    reach = round(EDGE_REACH * canvas.scale)
    first = edge_row(
        quiet,
        round(top),
        outward=-1,
        reach=reach,
        limit=math.floor(core[1]),
    )
    last = edge_row(
        quiet,
        round(bottom) - 1,
        outward=1,
        reach=reach,
        limit=math.ceil(core[3]),
    )
    cut = image.crop((0, first, image.width, last + 1))
    return cut, replace(canvas, top=canvas.top + first)


@dataclass(frozen=True, slots=True)
class _Job:
    """A render, the requests to mark on it, and its page.

    Attributes:
        image: The render, which no other job draws on.
        canvas: The pixels of the render.
        page: Position of the page, from 0.
        plans: The requests on the page whose marks it shows, if any.
        trim: Where a zoom ends, ``None`` for a page, which is shown whole.
    """

    image: Image.Image
    canvas: _Canvas
    page: int
    plans: tuple[_Planned, ...] = ()
    trim: _Trim | None = None

    def marked(self) -> Image.Image:
        """Return the render, cut if it is a zoom, with the marks on it."""
        image, canvas = self.image, self.canvas
        if self.trim is not None:
            image, canvas = _trimmed(image, canvas, self.trim)
        marks = _Marks()
        for planned in self.plans:
            planned.mark(marks)
        return marks.draw(image, canvas)


def _shot(job: _Job) -> Screenshot | None:
    """Return the screenshot of a job, marked and encoded."""
    try:
        image = job.marked()
    except ValueError:
        return None
    return encode_shot((image, job.page))


class _Flight:
    """The screenshots that a pool of threads marks and encodes.

    At most a given number of them wait or are being made: past it, the
    first one made, whichever it is, is waited for before another one is
    given to the pool, so that a page, much longer to encode than a zoom,
    does not hold the others back.
    """

    def __init__(self, pool: ThreadPoolExecutor, most: int) -> None:
        """Start with no screenshot.

        Args:
            pool: The threads that make the screenshots.
            most: Most screenshots that wait or are being made.
        """
        self._pool = pool
        self._most = most
        self._running: set[Future[Screenshot | None]] = set()

    def send(self, job: _Job | None) -> Future[Screenshot | None] | None:
        """Give the job of a screenshot to the pool, if there is one.

        Args:
            job: Its render and its marks; ``None`` for no screenshot.

        Returns:
            The screenshot to come; ``None`` without a job.
        """
        if job is None:
            return None
        if len(self._running) >= self._most:
            done = wait(self._running, return_when=FIRST_COMPLETED)
            self._running = done.not_done
        future = self._pool.submit(_shot, job)
        self._running.add(future)
        return future


def _made(future: Future[Screenshot | None] | None) -> Screenshot | None:
    """Return the screenshot of a future, ``None`` without one."""
    return None if future is None else future.result()


@dataclass(slots=True)
class _OpenPage:
    """A page of the PDF, open for its screenshots.

    Attributes:
        page: The page.
        crop: Its crop box, in the coordinates of PDFium.
        guard: The checks of its image objects, shared by its renders.
        text: Its characters, once they are asked for.
    """

    page: pdfium.PdfPage
    crop: Box
    guard: RenderGuard
    text: _PageText | None = field(default=None)

    def close(self) -> None:
        """Close the text page, if it was opened, then the page."""
        if self.text is not None:
            self.text.close()
            self.text = None
        self.page.close()


class PdfShooter:
    """Take the screenshots of a PDF.

    The PDF is opened at the first screenshot, and closed with the shooter,
    which is a context manager. The last `OPEN_PAGES` pages used stay open,
    with their checks and their characters: the screenshots are best taken
    page after page. A closed shooter takes no screenshot, and neither does
    a shooter whose PDF cannot be opened.
    """

    def __init__(self, path: Path) -> None:
        """Keep the PDF, which is opened at the first screenshot.

        Args:
            path: The PDF file.
        """
        self._path = path
        self._pdf: pdfium.PdfDocument | None = None
        self._pages: dict[int, _OpenPage] = {}
        self._closed = False

    def __enter__(self) -> Self:
        """Return the shooter.

        Returns:
            The shooter itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the PDF, if it was opened."""
        self.close()

    def close(self) -> None:
        """Close the PDF, if it was opened, and take no more screenshots."""
        self._closed = True
        for opened in self._pages.values():
            opened.close()
        self._pages.clear()
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    def _page(self, index: int) -> _OpenPage:
        """Return a page, open, opening the PDF first if needed.

        The page becomes the last one used; the one used longest ago is
        closed when more than `OPEN_PAGES` would stay open.

        Args:
            index: Position of the page, from 0.

        Returns:
            The page.

        Raises:
            OSError: If the PDF cannot be read; the shooter is then closed.
            PdfiumError: If PDFium cannot open the PDF, and the shooter is
                then closed, or if the PDF has no such page.
        """
        opened = self._pages.pop(index, None)
        if opened is None:
            if self._pdf is None:
                try:
                    self._pdf = pdfium.PdfDocument(self._path)
                except pdfium.PdfiumError, OSError:
                    self._closed = True
                    raise
            page = self._pdf[index]
            opened = _OpenPage(page, page.get_bbox(), RenderGuard(page))
            if len(self._pages) >= OPEN_PAGES:
                self._pages.pop(next(iter(self._pages))).close()
        self._pages[index] = opened
        return opened

    def _area(self, box: PageBox) -> Area:
        """Return the characters of a box of a page, its text read once."""
        opened = self._page(box.page)
        if opened.text is None:
            opened.text = _PageText(opened.page, opened.crop, box.turn)
        return Area(box.page, opened.text.inside(box.edges))

    def _spot(self, request: ShotRequest) -> Spot | None:
        """Return where the part of the text of a request lies, if found."""
        if request.mode != "find" or request.wanted is None:
            return None
        areas = [self._area(box) for box in request.boxes]
        return locate(areas, request.wanted)

    def _plan(self, request: ShotRequest) -> _Planned:
        """Return the plan of a request, raising what stops it.

        The page is that of the part of the text, if it is found, else that
        of the first box.
        """
        spot = self._spot(request)
        page = request.boxes[0].page if spot is None else spot.page
        boxes = [box for box in request.boxes if box.page == page]
        turn = boxes[0].turn
        size = upright_size(self._page(page).crop, turn)
        edges = tuple(box.edges for box in boxes)
        return _Planned(request, page, turn, size, edges, spot)

    def _planned(self, request: ShotRequest) -> _Planned | None:
        """Return the plan of a request, if it can be shown.

        Returns:
            ``None`` if the shooter is closed, if the boxes are not valid, or
            if their page cannot be read.
        """
        if self._closed or not _valid(request.boxes):
            return None
        try:
            return self._plan(request)
        except pdfium.PdfiumError, OSError, ValueError:
            return None

    def _job(  # ruff: ignore[too-many-arguments] - the page, its area, its marks
        self,
        page: int,
        turn: int,
        area: Box,
        *,
        scale: float,
        plans: Sequence[_Planned] = (),
        trim: _Trim | None = None,
    ) -> _Job | None:
        """Return the render of an area of a page, to mark with requests.

        Args:
            page: Position of the page, from 0.
            turn: The quarter turns, clockwise, of the text of the page.
            area: The area, in points of the page turned upright.
            scale: Pixels for each point that the render would rather have,
                lowered as `fitted_scale` tells.
            plans: The requests on the page to mark, if any.
            trim: Where a zoom ends, ``None`` for a page.

        Returns:
            The render and the requests to mark on it; ``None`` if the area
            cannot be rendered.
        """
        scale = fitted_scale(area, scale)
        try:
            opened = self._page(page)
            image = render_upright(
                opened.page,
                area,
                turn=turn,
                guard=opened.guard,
                scale=scale,
            )
        except UnreadableImage, pdfium.PdfiumError, OSError, ValueError:
            return None
        height = upright_size(opened.crop, turn)[1]
        canvas = _Canvas.of_area(area, height, scale)
        return _Job(image, canvas, page, tuple(plans), trim)

    def _zoom(self, planned: _Planned) -> _Job | None:
        """Return the render of the zoom of a request, and its marks.

        Returns:
            The area that `zoom_area` gives, and `EDGE_REACH` points above
            and under it, rendered at `ZOOM_SCALE`, to be cut as `_trimmed`
            tells; ``None`` if the area is not on the page, or if it cannot
            be rendered.
        """
        marks = () if planned.spot is None else planned.spot.marks
        area = zoom_area(planned.blocks, marks, planned.size)
        if area is None:
            return None
        left, bottom, right, top = area
        reached = (
            left,
            max(bottom - EDGE_REACH, 0.0),
            right,
            min(top + EDGE_REACH, planned.size[1]),
        )
        return self._job(
            planned.page,
            planned.turn,
            reached,
            scale=ZOOM_SCALE,
            plans=(planned,),
            trim=_Trim(area, _core(planned.blocks, marks)),
        )

    def _whole(
        self,
        page: int,
        turn: int,
        plans: Sequence[_Planned] = (),
    ) -> _Job | None:
        """Return the render of a whole page, and the marks of its requests.

        Args:
            page: Position of the page, from 0.
            turn: The quarter turns, clockwise, of the text of the page.
            plans: The requests on the page to mark, if any.

        Returns:
            The page, rendered at `PAGE_SCALE`; ``None`` if the shooter is
            closed, or if the page cannot be rendered.
        """
        if self._closed:
            return None
        try:
            size = upright_size(self._page(page).crop, turn)
        except pdfium.PdfiumError, OSError:
            return None
        whole = (0.0, 0.0, *size)
        return self._job(page, turn, whole, scale=PAGE_SCALE, plans=plans)

    def _shown(self, plans: Sequence[_Planned]) -> _Job | None:
        """Return the render of the page of some requests, and their marks.

        Args:
            plans: Requests on one page, at least one.

        Returns:
            The page, as `_whole` renders it, with their marks.
        """
        first = plans[0]
        return self._whole(first.page, first.turn, plans)

    def render(self, request: ShotRequest) -> tuple[Image.Image, int] | None:
        """Return the zoom of a request, before it is encoded.

        Args:
            request: What the screenshot shows.

        Returns:
            The zoom, in RGBX, and the position of its page, from 0: the
            area that `zoom_area` gives on the page of the part of the text,
            if it is found, else on the page of the first box, its top and
            its bottom on rows without ink, upright, with its marks. ``None``
            if the shooter is closed, if the boxes are not valid, or if the
            area cannot be rendered.
        """
        planned = self._planned(request)
        job = None if planned is None else self._zoom(planned)
        if job is None:
            return None
        return job.marked(), job.page

    def shoot_all(self, requests: Sequence[ShotRequest]) -> Shots:
        """Return the zooms of requests, and the pages that they mark.

        The requests are taken by the first page of their boxes, as no
        request marks a page before it: a page is shot once no request left
        can mark it. The zooms and the pages are rendered here, each page
        once, and the screenshots are marked and encoded in a thread for
        each processor: PDFium runs in this thread only, while Pillow and the
        encoder of WebP run outside of the lock of the interpreter. At most
        `SHOTS_PER_WORKER` screenshots for each thread wait.

        Args:
            requests: What each screenshot shows.

        Returns:
            The zoom of each request, as `encode_shot` encodes the zoom of
            `render`, and the page that it marks, whole, with the marks of
            all the requests on it.
        """
        workers = os.process_cpu_count() or 1
        firsts = [_first_page(request) for request in requests]
        zooms: list[Future[Screenshot | None] | None] = [None] * len(requests)
        page_of: list[int | None] = [None] * len(requests)
        pages: dict[int, Future[Screenshot | None] | None] = {}
        waiting: dict[int, list[_Planned]] = {}
        with ThreadPoolExecutor(workers) as pool:
            flight = _Flight(pool, SHOTS_PER_WORKER * workers)
            for index in sorted(range(len(requests)), key=firsts.__getitem__):
                for page in sorted(p for p in waiting if p < firsts[index]):
                    pages[page] = flight.send(self._shown(waiting.pop(page)))
                planned = self._planned(requests[index])
                if planned is None:
                    continue
                page_of[index] = planned.page
                waiting.setdefault(planned.page, []).append(planned)
                zooms[index] = flight.send(self._zoom(planned))
            for page, plans in waiting.items():
                pages[page] = flight.send(self._shown(plans))
        shots = {page: _made(future) for page, future in pages.items()}
        return Shots(
            tuple(_made(future) for future in zooms),
            tuple(None if page is None else shots[page] for page in page_of),
        )

    def shoot_pages(
        self,
        pages: Sequence[tuple[int, int]],
    ) -> tuple[Screenshot | None, ...]:
        """Return the screenshots of whole pages, without marks.

        The pages are rendered here, and encoded in a thread for each
        processor, as `shoot_all` does.

        Args:
            pages: The position of each page, from 0, and the quarter turns,
                clockwise, of its text.

        Returns:
            The screenshot of each page, in their order, as `shoot_all`
            shoots a page, but without marks; ``None`` for a page that
            cannot be rendered.
        """
        workers = os.process_cpu_count() or 1
        with ThreadPoolExecutor(workers) as pool:
            flight = _Flight(pool, SHOTS_PER_WORKER * workers)
            sent = [flight.send(self._whole(*page)) for page in pages]
        return tuple(_made(future) for future in sent)


def encode_shot(shot: tuple[Image.Image, int] | None) -> Screenshot | None:
    """Return a screenshot encoded in lossless WebP.

    Args:
        shot: The screenshot and the position of its page, as
            `PdfShooter.render` gives them, if any.

    Returns:
        The screenshot, encoded at `SHOT_EFFORT` with `SHOT_METHOD`; ``None``
        without one or if it cannot be encoded.
    """
    if shot is None:
        return None
    image, page = shot
    try:
        encoded = encode_webp(
            image,
            quality=SHOT_EFFORT,
            method=SHOT_METHOD,
            lossless=True,
        )
    except OSError, ValueError:
        return None
    return Screenshot(encoded, page + 1)
