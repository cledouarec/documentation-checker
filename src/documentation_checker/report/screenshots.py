# Copyright 2026 Christophe Le Douarec
"""The screenshots of the sides of the findings, made once for both reports.

When the screenshots are asked for, each side of a finding that comes from a
PDF, and whose source tells where its blocks lie, shows a zoom on the place
of its change, and the reports show each page that these sides mark, whole,
with the marks of all of them, as `PdfShooter` takes them. A comparison
shows its pages side by side, as `page_rows` lays them out: each page of the
reference beside the page of the new document that `matched_pages` matches
with it, which is shown without marks if no finding marks it. Each finding
of the reports gets its screenshots, and so does each occurrence that a
group lists, unless the configuration limits them to the first ones, in the
order of the reports. A side with an image keeps its thumbnail instead.
"""

import bisect
from collections import Counter
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import TYPE_CHECKING, Literal, cast

from documentation_checker.audit.engine import AuditResult
from documentation_checker.compare.pages import heaviest_chain
from documentation_checker.images.locate import Wanted
from documentation_checker.images.screenshot import PdfShooter, ShotRequest
from documentation_checker.report.context import (
    ReportContext,
    columns,
    report_order,
    side_of,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence

    from documentation_checker.compare.engine import CompareResult
    from documentation_checker.images.screenshot import Screenshot, ShotStyle
    from documentation_checker.images.thumbnail import Thumbnail
    from documentation_checker.model.document import Document
    from documentation_checker.model.findings import Finding, Side
    from documentation_checker.report.context import Column

# The column of a document in the reports.
type Which = Literal["ref", "new"]


@dataclass(frozen=True, slots=True)
class PageShot:
    """The screenshot of a page, with the marks of all its findings.

    Attributes:
        column: The column of the document of the page.
        page: Number of the page, from 1.
        image: The screenshot of the whole page, in WebP.
        findings: The findings and the occurrences whose sides it marks, in
            the order of the reports.
    """

    column: Column
    page: int
    image: Thumbnail
    findings: tuple[Finding, ...]

    @property
    def title(self) -> str:
        """Title of the page in the reports, such as ``Reference, page 9``."""
        return f"{self.column.label}, page {self.page}"

    @property
    def anchor(self) -> str:
        """Identifier of the page in the HTML report, such as ``p-ref-9``."""
        return f"p-{self.column.which}-{self.page}"


@dataclass(frozen=True, slots=True)
class PageRow:
    """Pages that the reports show side by side, one for each column.

    Attributes:
        ref: The page of the reference, or of the audited document, if any.
        new: The page of the new document beside it, if any.
    """

    ref: PageShot | None
    new: PageShot | None = None

    def of(self, which: Which) -> PageShot | None:
        """Return the page of a column, if the row has one."""
        return self.ref if which == "ref" else self.new

    @property
    def title(self) -> str:
        """Title of the row, such as ``Reference, page 9 · New, page 8``."""
        pages = (self.ref, self.new)
        return " \u00b7 ".join(
            page.title for page in pages if page is not None
        )


@dataclass(slots=True)
class _Made:
    """The screenshots once they are taken.

    Attributes:
        zooms: The zoom of each side, by the identity of its finding and its
            column.
        page_of: The page of each side, the same way.
        rows: The pages, side by side, in the order of the documents.
    """

    zooms: dict[tuple[int, Which], Screenshot | None] = field(
        default_factory=dict,
    )
    page_of: dict[tuple[int, Which], PageShot] = field(default_factory=dict)
    rows: list[PageRow] = field(default_factory=list)


def matched_pages(
    links: Mapping[tuple[int, int], float],
    sides: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return the pages of the reference matched with those of the new one.

    Args:
        links: How much text links each page of the reference with each page
            of the new document, by their numbers, as `page_links` counts it.
        sides: The pages of the two sides of each finding that has both.

    Returns:
        The heaviest chain of the links, as `heaviest_chain` finds it, where
        a link of a finding weighs more than all the text: the two sides of
        as many findings as can be face each other, and among such chains
        the pages that share the most text.
    """
    found = Counter(sides)
    unit = sum(links.values()) + 1
    return heaviest_chain(
        {
            pair: found[pair] * unit + links.get(pair, 0)
            for pair in found.keys() | links.keys()
        },
    )


def page_rows(
    matched: Sequence[tuple[int, int]],
    ref: Collection[int],
    new: Collection[int],
) -> list[tuple[int | None, int | None]]:
    """Return the pages to show side by side, in the order of the documents.

    Args:
        matched: The pages of the reference matched with those of the new
            document, in the order of both, as `matched_pages` matches them.
        ref: The numbers of the pages of the reference to show.
        new: The numbers of the pages of the new document to show.

    Returns:
        The rows of a page of the reference and a page of the new document:
        each matched pair of which one page is to show, the other page
        shown beside it; and each page to show that no pair holds, between
        the pairs before it and after it in its document, beside a page of
        the other document that falls there too, in their order, if there is
        one, else alone.
    """
    refs = [ref_page for ref_page, _ in matched]
    news = [new_page for _, new_page in matched]
    alone_ref: dict[int, list[int]] = {}
    for page in sorted(set(ref) - set(refs)):
        alone_ref.setdefault(bisect.bisect(refs, page), []).append(page)
    alone_new: dict[int, list[int]] = {}
    for page in sorted(set(new) - set(news)):
        alone_new.setdefault(bisect.bisect(news, page), []).append(page)
    rows: list[tuple[int | None, int | None]] = []
    for gap in range(len(matched) + 1):
        rows += zip_longest(alone_ref.get(gap, ()), alone_new.get(gap, ()))
        if gap < len(matched) and (refs[gap] in ref or news[gap] in new):
            rows.append(matched[gap])
    return rows


def _turns(document: Document) -> dict[int, int]:
    """Return the quarter turns of the text of the pages of a document.

    Args:
        document: A document read with the boxes of its blocks.

    Returns:
        The turns of each page that a block lies on, by its position, from
        0, as the boxes of its blocks tell them.
    """
    return {
        box.page: box.turn
        for block in document.blocks
        for box in block.source.boxes
    }


def shot_request(
    side: Side | None,
    plain: ShotStyle = "block",
) -> ShotRequest | None:
    """Return what the screenshot of a side of a finding shows.

    Args:
        side: A side of a finding, if any.
        plain: The colors of a side without a changed segment: ``"block"``
            in a comparison, whose blocks then kept their text but not their
            place or their kind, ``"mark"`` in an audit, whose finding then
            points at the blocks.

    Returns:
        ``None`` without a side, for a side with an image, or for one whose
        source has no boxes. Else, for a side without a changed segment, its
        blocks tinted whole in the colors of ``plain``; for a side whose one
        segment is changed, its blocks tinted whole; else its changed part,
        found among the text around it and marked, with a caret where it is
        empty. The marks of a change have its colors.
    """
    if side is None or side.image is not None or not side.source.boxes:
        return None
    boxes = side.source.boxes
    segments = side.segments
    changed = [
        i for i, segment in enumerate(segments) if segment.op != "equal"
    ]
    if not changed:
        return ShotRequest(boxes, "whole", plain)
    first, last = changed[0], changed[-1]
    style = cast("ShotStyle", segments[first].op)
    if len(segments) == 1:
        return ShotRequest(boxes, "whole", style)
    wanted = Wanted(
        "".join(segment.text for segment in segments[:first]),
        "".join(segment.text for segment in segments[first : last + 1]),
        "".join(segment.text for segment in segments[last + 1 :]),
    )
    return ShotRequest(boxes, "find", style, wanted)


def _listed(findings: Sequence[Finding]) -> list[Finding]:
    """Return the findings in the order of the reports, groups as they list.

    Args:
        findings: The findings of a result.

    Returns:
        Each finding, in the order of `report_order`, or for a group the
        occurrences that the reports list, as `ReportContext.shown` keeps
        them.
    """
    return [
        item
        for finding in report_order(findings)
        for item in (
            ReportContext.shown(finding) if finding.occurrences else (finding,)
        )
    ]


class Screenshots:
    """The screenshots of the sides of the findings, made once for both.

    What each screenshot shows is chosen when this object is made: for the
    findings that the reports list, as `_listed` gives them, with a side to
    show in a PDF, up to the limit. The screenshots are all taken at the
    first demand, one document after the other, page after page, by a
    `PdfShooter` that is closed once they are taken.
    """

    def __init__(
        self,
        result: CompareResult | AuditResult,
        *,
        enabled: bool = True,
        limit: int | None = None,
    ) -> None:
        """Choose the findings whose sides get a screenshot.

        Args:
            result: Result of a comparison or of an audit.
            enabled: Whether the reports show screenshots at all.
            limit: Most findings and occurrences that get screenshots, the
                first ones in the order of the reports; ``None`` for all.
        """
        # The findings chosen, in the order of the reports, with what each
        # of their sides shows, by its column.
        self._chosen: list[tuple[Finding, dict[Which, ShotRequest]]] = []
        self._pdfs: list[Column] = []
        self._links: Mapping[tuple[int, int], float] = (
            {} if isinstance(result, AuditResult) else result.page_links
        )
        self._left_out = 0
        self._made: _Made | None = None
        pdfs = [c for c in columns(result) if c.document.format == "pdf"]
        if not enabled or not pdfs:
            return
        plain: ShotStyle = (
            "mark" if isinstance(result, AuditResult) else "block"
        )
        chosen = []
        for item in _listed(result.findings):
            requests = {
                column.which: request
                for column in pdfs
                if (
                    request := shot_request(side_of(item, column.which), plain)
                )
            }
            if requests:
                chosen.append((item, requests))
        if limit is not None:
            self._left_out = max(len(chosen) - limit, 0)
            chosen = chosen[:limit]
        self._chosen = chosen
        self._pdfs = pdfs

    def get(self, finding: Finding, which: Which) -> Screenshot | None:
        """Return the zoom on the place of a side of a finding.

        Args:
            finding: A finding, or an occurrence of a group.
            which: The column of the side.

        Returns:
            The zoom, the same one each time; ``None`` for a finding that
            was not chosen, for a side without screenshot, or for one whose
            place cannot be rendered.
        """
        return self._take().zooms.get((id(finding), which))

    def page(self, finding: Finding, which: Which) -> PageShot | None:
        """Return the page of a side of a finding, with all its findings.

        Args:
            finding: A finding, or an occurrence of a group.
            which: The column of the side.

        Returns:
            The page that the side marks; ``None`` for a finding that was not
            chosen, for a side without screenshot, or for one whose page
            cannot be rendered.
        """
        return self._take().page_of.get((id(finding), which))

    def rows(self) -> list[PageRow]:
        """Return the pages of the findings, side by side.

        Returns:
            Each page that the sides of the findings mark, once, with the
            marks of all of them, beside the page of the other document that
            matches it, as `page_rows` lays them out, in the order of the
            documents.
        """
        return list(self._take().rows)

    def _take(self) -> _Made:
        """Return the screenshots, taken at the first demand."""
        if self._made is None:
            self._made = self._shoot()
        return self._made

    def _shoot(self) -> _Made:
        """Return the screenshots of the chosen findings, page after page.

        Findings whose sides ask for the same screenshot share it. The pages
        that the rows show beside the others, and that no finding marks, are
        shot then, without marks.
        """
        made = _Made()
        # The screenshot of each page and its findings, and the pages of the
        # sides of each finding, by the columns.
        pages: dict[Which, dict[int, tuple[Screenshot, list[Finding]]]] = {}
        sides: dict[int, dict[Which, int]] = {}
        for column in self._pdfs:
            on_page = pages[column.which] = {}
            for item, zoom, shot in self._shot_sides(column):
                made.zooms[id(item), column.which] = zoom
                if shot is not None:
                    on_page.setdefault(shot.page, (shot, []))[1].append(item)
                    sides.setdefault(id(item), {})[column.which] = shot.page
        matched = matched_pages(
            self._links,
            [
                (pair["ref"], pair["new"])
                for pair in sides.values()
                if len(pair) > 1
            ],
        )
        rows = page_rows(matched, pages.get("ref", {}), pages.get("new", {}))
        for column in self._pdfs:
            self._shoot_beside(column, rows, pages[column.which])
        shown = {
            (column.which, number): PageShot(
                column,
                number,
                shot.image,
                tuple(items),
            )
            for column in self._pdfs
            for number, (shot, items) in pages[column.which].items()
        }
        for ref_page, new_page in rows:
            row = PageRow(
                shown.get(("ref", ref_page)),
                shown.get(("new", new_page)),
            )
            if row.ref is not None or row.new is not None:
                made.rows.append(row)
        for (which, _), page in shown.items():
            for item in page.findings:
                made.page_of[id(item), which] = page
        return made

    def _shot_sides(
        self,
        column: Column,
    ) -> list[tuple[Finding, Screenshot | None, Screenshot | None]]:
        """Return the zoom and the page of each side of a column.

        Args:
            column: A column of a PDF.

        Returns:
            Each chosen finding with a side in the column, in the order of the
            reports, with the zoom of its side and the page that it marks.
        """
        which = column.which
        jobs = [
            (item, sides[which])
            for item, sides in self._chosen
            if which in sides
        ]
        requests = list(dict.fromkeys(request for _, request in jobs))
        with PdfShooter(column.document.path) as shooter:
            shots = shooter.shoot_all(requests)
        position = {request: i for i, request in enumerate(requests)}
        return [
            (
                item,
                shots.zooms[position[request]],
                shots.pages[position[request]],
            )
            for item, request in jobs
        ]

    @staticmethod
    def _shoot_beside(
        column: Column,
        rows: Sequence[tuple[int | None, int | None]],
        on_page: dict[int, tuple[Screenshot, list[Finding]]],
    ) -> None:
        """Shoot the pages of a column that the rows show and no finding marks.

        Args:
            column: A column of a PDF.
            rows: The numbers of the pages of each row, as `page_rows` gives
                them.
            on_page: The screenshot of each page of the column and its
                findings, by its number, which the pages shot join, without
                findings.
        """
        side = 0 if column.which == "ref" else 1
        wanted = sorted(
            {number for row in rows if (number := row[side]) is not None}
            - on_page.keys(),
        )
        if not wanted:
            return
        turns = _turns(column.document)
        pages = [(number - 1, turns.get(number - 1, 0)) for number in wanted]
        with PdfShooter(column.document.path) as shooter:
            shots = shooter.shoot_pages(pages)
        for number, shot in zip(wanted, shots, strict=True):
            if shot is not None:
                on_page[number] = (shot, [])

    def note(self) -> str | None:
        """Return what the reports say of the findings left without screenshot.

        Returns:
            A sentence when the limit left findings out, else ``None``.
        """
        if not self._left_out:
            return None
        return (
            f"Only the first {len(self._chosen)} findings have screenshots, "
            f"as screenshots_max sets: {self._left_out} more have none."
        )
