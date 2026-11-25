# Copyright 2026 Christophe Le Douarec
"""Markdown report of a comparison or of an audit.

The thumbnails and the screenshots that the report shows are WebP files in a
directory next to it, named by their content. A file already there is kept,
and none is ever deleted. Each page that the screenshots show is in the report
once, in a table of the Pages section, beside the page of the other document
that matches it, after an anchor that the findings on the page link to.
"""

import difflib
import re
from typing import TYPE_CHECKING, Final, Literal

from documentation_checker.compare.aggregate import shown_pattern
from documentation_checker.report.context import (
    CATEGORY_TITLES,
    SEVERITY_TITLES,
    TEMPLATES,
    UNKNOWN,
    Column,
    ReportContext,
    Thumbnails,
    counted,
    is_code,
    side_of,
)
from documentation_checker.report.screenshots import PageShot, Screenshots

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from documentation_checker.audit.engine import AuditResult
    from documentation_checker.compare.engine import CompareResult
    from documentation_checker.images.thumbnail import Thumbnail
    from documentation_checker.model.findings import DiffSegment, Finding, Side

# The directory of the thumbnails and of the screenshots, next to the report.
ASSETS_DIR: Final = "report_assets"
_ABSENT: Final = "\U00002014"
_MARKUP: Final = {
    "delete": ("<del>", "</del>"),
    "insert": ("<ins>", "</ins>"),
    "mark": ("<mark>", "</mark>"),
    "equal": ("", ""),
}
# The characters of HTML and of the inline syntax of Markdown.
_ESCAPES: Final = str.maketrans(
    {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        **{char: f"\\{char}" for char in "\\`*_[]|~"},
    },
)
_LINE_BREAK_RE: Final = re.compile(r"\r\n|[\r\n]")
# What opens a heading, a bullet or a thematic break, then what opens a
# numbered list, after the blanks that a line may start with.
_OPENING_MARK_RE: Final = re.compile(r"^([ \t]*)([#+-])")
_OPENING_NUMBER_RE: Final = re.compile(r"^([ \t]*\d+)([.)])")
_BACKTICKS_RE: Final = re.compile(r"`+")
_PIPE_RE: Final = re.compile(r"(\\*)\|")
_BACKSLASH: Final = "\\"


def _backtick_fence(text: str, minimum: int = 1) -> str:
    """Return backticks that are more than any run of backticks in a text."""
    longest = max(map(len, _BACKTICKS_RE.findall(text)), default=0)
    return "`" * max(longest + 1, minimum)


def md_fence(lines: Sequence[str]) -> str:
    """Return the fence of a block of code in Markdown.

    Args:
        lines: Lines of the block.

    Returns:
        Backticks that are more than any run of backticks in the lines, and
        three at least.
    """
    return _backtick_fence("\n".join(lines), minimum=3)


def md_text(text: str) -> str:
    """Return a text of a document as safe text in a line of Markdown.

    The result is one line that cannot start a tag, an emphasis, a code span,
    a link, a table cell, a heading or a list. It still holds the controls of
    the text: see `printable` for them.

    Args:
        text: Any text that comes from a document.

    Returns:
        The text with each line break turned into a space, the ampersand and
        the angle brackets turned into entities, and a backslash put before
        each backslash, backtick, asterisk, underscore, square bracket, pipe
        and tilde. A backslash also comes before a ``#``, a ``-`` or a ``+``
        that opens the text, and before the ``.`` or the ``)`` of a number
        that opens it, whether or not blanks come first. The parentheses are
        left as they are.
    """
    text = _LINE_BREAK_RE.sub(" ", text).translate(_ESCAPES)
    text = _OPENING_MARK_RE.sub(r"\1\\\2", text)
    return _OPENING_NUMBER_RE.sub(r"\1\\\2", text)


def md_code(text: str) -> str:
    """Return a text of a document as a Markdown code span.

    Args:
        text: Any text that comes from a document, such as a path.

    Returns:
        The text on one line between fences of backticks, which are one more
        than the longest run of backticks in the text, and with a space inside
        each fence if the text holds a backtick. In a table cell, use
        `md_cell_code`.
    """
    text = _LINE_BREAK_RE.sub(" ", text)
    fence = _backtick_fence(text)
    blank = " " if "`" in text else ""
    return f"{fence}{blank}{text}{blank}{fence}"


def _hide_pipe(match: re.Match[str]) -> str:
    """Return a pipe with the backslashes that keep it in its table cell."""
    run = match.group(1)
    return run + _BACKSLASH * (1 + len(run) % 2) + "|"


def md_cell_code(text: str) -> str:
    """Return a text of a document as a code span in a Markdown table cell.

    A table ends a cell at each pipe, even in a code span, unless an odd number
    of backslashes come right before it, and the cell then loses the last one.
    So a pipe gets one more backslash, or two more if an odd number of them
    come before it already, in which case the span shows one backslash more
    than the text.

    Args:
        text: Any text that comes from a document, such as a path.

    Returns:
        The code span of `md_code`, whose pipes cannot end the cell.
    """
    return _PIPE_RE.sub(_hide_pipe, md_code(text))


def md_cell(segments: Sequence[DiffSegment]) -> str:
    """Return diff segments as the content of a Markdown table cell.

    Args:
        segments: Unchanged, deleted, inserted and marked pieces of text.

    Returns:
        Escaped text, with a ``<br>`` for each line break, deletions in
        ``<del>``, insertions in ``<ins>`` and marked pieces in ``<mark>``.
        Empty pieces, such as the place of a text that only the other side
        has, are left out.
    """
    parts = []
    for segment in segments:
        if not segment.text:
            continue
        opening, closing = _MARKUP[segment.op]
        lines = _LINE_BREAK_RE.split(segment.text)
        text = "<br>".join(md_text(line) for line in lines)
        parts.append(f"{opening}{text}{closing}")
    return "".join(parts)


def diff_lines(finding: Finding) -> list[str]:
    """Return the lines that tell how the code of a finding changed.

    Args:
        finding: A finding with a code side.

    Returns:
        The lines of both sides, each after a two character prefix: two
        blanks for a line that both sides have, ``- `` for a line of the
        reference only and ``+ `` for a line of the new side only. Where
        lines changed, those of the reference come first. A missing side
        counts for no line.
    """
    ref = finding.ref.text.splitlines() if finding.ref else []
    new = finding.new.text.splitlines() if finding.new else []
    matcher = difflib.SequenceMatcher(None, ref, new, autojunk=False)
    lines: list[str] = []
    for tag, ref_start, ref_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            lines += [f"  {line}" for line in ref[ref_start:ref_end]]
            continue
        lines += [f"- {line}" for line in ref[ref_start:ref_end]]
        lines += [f"+ {line}" for line in new[new_start:new_end]]
    return lines


def _plain(value: str) -> str:
    """Return a raw value of the inputs table, as text of a cell."""
    return value if value == UNKNOWN else md_text(value)


def _source(
    context: ReportContext,
    side: Side | None,
    which: Literal["ref", "new"],
) -> str:
    """Return the Markdown link to the source of a side."""
    source = context.source(side, which)
    if source is None:
        return _ABSENT
    href, url = source
    return f"[{md_text(href)}]({url})"


def _cell(side: Side | None) -> str:
    """Return the table cell of a side."""
    return _ABSENT if side is None else md_cell(side.segments)


def _count_of(categories: Sequence[tuple[object, Sequence[object]]]) -> int:
    """Return the number of findings in the categories of a severity."""
    return sum(len(findings) for _, findings in categories)


def _holds(target: Path, size: int) -> bool:
    """Tell whether a file exists with a given size.

    Args:
        target: The file.
        size: The size it must have, in bytes.

    Returns:
        Whether it is a file of that size.
    """
    try:
        return target.is_file() and target.stat().st_size == size
    except OSError:
        return False


class _Assets:
    """The images that a Markdown report shows, as files next to it.

    The images are the thumbnails of the images of the sides of the findings
    and the screenshots of their pages.
    """

    def __init__(
        self,
        thumbnails: Thumbnails,
        screenshots: Screenshots,
        columns: Sequence[Column],
    ) -> None:
        """Start with no image shown.

        Args:
            thumbnails: Thumbnails of the sides of the findings.
            screenshots: Screenshots of the sides of the findings.
            columns: The columns of the report, which name the images.
        """
        self._thumbnails = thumbnails
        self._screenshots = screenshots
        self._columns = tuple(columns)
        self._labels = {
            column.which: f"{column.label} image" for column in columns
        }
        self._shown: dict[str, Thumbnail] = {}

    def __call__(self, side: Side | None, which: Literal["ref", "new"]) -> str:
        """Return the image that opens the table cell of a side.

        Args:
            side: A side of a finding, if any.
            which: The document of the side.

        Returns:
            The image of its thumbnail and a line break, or "" without one.
        """
        thumb = self._thumbnails.get(side, which)
        if thumb is None:
            return ""
        return f"{self._embed(self._labels[which], thumb)}<br>"

    def _embed(self, label: str, image: Thumbnail) -> str:
        """Return the Markdown image of a file to write next to the report."""
        name = image.name
        self._shown.setdefault(name, image)
        return f"![{label}]({ASSETS_DIR}/{name})"

    def _screenshots_of(self, finding: Finding, column: Column) -> list[str]:
        """Return the zoom of a side and the link to its page, if any."""
        shown = []
        zoom = self._screenshots.get(finding, column.which)
        if zoom is not None:
            label = f"{column.label}, page {zoom.page}"
            shown.append(self._embed(label, zoom.image))
        page = self._screenshots.page(finding, column.which)
        if page is not None:
            count = counted(len(page.findings), "finding")
            link = f"[Whole page {page.page}](#{page.anchor})"
            shown.append(f"{link} ({count})")
        return shown

    def shot(self, finding: Finding, column: Column) -> str:
        """Return the screenshots that end the table cell of a side.

        Args:
            finding: A finding, or an occurrence of a group.
            column: The column of the side.

        Returns:
            The image of its zoom and the link to its page, each after a
            line break; "" without any.
        """
        shown = self._screenshots_of(finding, column)
        return "".join(f"<br>{part}" for part in shown)

    def shot_line(self, finding: Finding) -> str:
        """Return the screenshots of the sides of a finding, on one line.

        Args:
            finding: A finding.

        Returns:
            The image of the zoom of each side and the link to its page, in
            the order of the columns, joined by a blank; "" without any.
        """
        return " ".join(
            part
            for column in self._columns
            for part in self._screenshots_of(finding, column)
        )

    def page_cell(self, page: PageShot | None) -> str:
        """Return the table cell of a page of the Pages section.

        Args:
            page: The page, if the row has one in the column.

        Returns:
            The image of its screenshot, then its findings, one on each line;
            a dash without a page.
        """
        if page is None:
            return _ABSENT
        label = (
            f"{page.title}, with its findings" if page.findings else page.title
        )
        lines = [self._embed(label, page.image)]
        lines += [
            f"{md_text(item.title)} ({md_cell_code(item.rule)}), "
            f"ID: {md_cell_code(item.id)}"
            for item in page.findings
        ]
        return "<br>".join(lines)

    def write(self, directory: Path) -> None:
        """Write each thumbnail shown that the directory does not hold yet.

        A file of the name and of another size, which a run cut short may
        leave, is written again; the bytes of a thumbnail are the same at
        each run.

        Args:
            directory: Where the thumbnails go, created if some are shown.
        """
        if not self._shown:
            return
        directory.mkdir(exist_ok=True)
        for name, thumb in self._shown.items():
            target = directory / name
            if not _holds(target, len(thumb.data)):
                target.write_bytes(thumb.data)


def render_markdown(
    result: CompareResult | AuditResult,
    path: Path,
    thumbnails: Thumbnails | None = None,
    screenshots: Screenshots | None = None,
) -> Path:
    """Write the Markdown report of a comparison or an audit, and its images.

    Args:
        result: Result of a comparison or of an audit.
        path: Report file to write.
        thumbnails: Thumbnails of the sides of the findings, shared with the
            other report; ``None`` for thumbnails of this report only.
        screenshots: Screenshots of the sides of the findings, shared with
            the other report; ``None`` for no screenshots.

    Returns:
        The written file. The thumbnails and the screenshots that it shows
        are in `ASSETS_DIR`, next to it.
    """
    if thumbnails is None:
        with Thumbnails(result) as own:
            return render_markdown(result, path, own, screenshots)
    if screenshots is None:
        screenshots = Screenshots(result, enabled=False)
    context = ReportContext(result, path.parent)
    assets = _Assets(thumbnails, screenshots, context.columns)
    inputs = [
        (label, tuple(map(md_cell_code if label == "Path" else _plain, row)))
        for label, row in context.inputs()
    ]
    text = TEMPLATES.get_template("report.md.j2").render(
        command=context.command,
        columns=context.columns,
        inputs=inputs,
        **context.headline(),
        categories=context.categories(),
        summary=[
            (SEVERITY_TITLES[severity], counts, total)
            for severity, counts, total in context.summary()
        ],
        normalization=context.normalization(),
        chrome=context.chrome(),
        warnings=context.warnings(),
        sections=[
            (SEVERITY_TITLES[severity], _count_of(categories), categories)
            for severity, categories in context.sections()
        ],
        category_titles=CATEGORY_TITLES,
        context=context,
        is_code=is_code,
        diff_lines=diff_lines,
        md_fence=md_fence,
        md_code=md_code,
        md_text=md_text,
        shown_pattern=shown_pattern,
        link=lambda side, which: _source(context, side, which),
        side_of=side_of,
        cell=_cell,
        image=assets,
        shot=assets.shot,
        shot_line=assets.shot_line,
        screenshot_note=screenshots.note(),
        rows=screenshots.rows(),
        page_cell=assets.page_cell,
    )
    assets.write(path.parent / ASSETS_DIR)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
