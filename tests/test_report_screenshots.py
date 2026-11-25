# Copyright 2026 Christophe Le Douarec
"""Tests for the screenshots of the sides of the findings in the reports."""

import math
import re
from dataclasses import replace
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING

import lxml.html
import pytest
from reportlab.lib.pagesizes import A4, A5

from documentation_checker.compare.engine import CompareResult, CompareStats
from documentation_checker.config import Config
from documentation_checker.images.locate import Wanted
from documentation_checker.images.screenshot import (
    PAGE_SCALE,
    ZOOM_MIN_WIDTH,
    ZOOM_SCALE,
    ShotRequest,
)
from documentation_checker.model.document import PageBox, SourceRef
from documentation_checker.model.findings import (
    Category,
    DiffSegment,
    Finding,
    Severity,
    Side,
)
from documentation_checker.pipeline import run_audit, run_compare
from documentation_checker.report.context import report_order
from documentation_checker.report.html import render_html
from documentation_checker.report.markdown import ASSETS_DIR, render_markdown
from documentation_checker.report.screenshots import (
    PageRow,
    PageShot,
    Screenshots,
    matched_pages,
    page_rows,
    shot_request,
)
from tests.builders import SAMPLE_EN, make_document, replace_text
from tests.pdf_builders import (
    build_fpdf2_pdf,
    build_lines_pdf,
    build_listing_pdf,
    build_reportlab_pdf,
)

if TYPE_CHECKING:
    from documentation_checker.audit.engine import AuditResult
    from documentation_checker.compare.drafts import SegmentOp

BOX = PageBox(0, 72.0, 600.0, 520.0, 640.0)
SOURCE = SourceRef("doc.pdf", page=1, boxes=(BOX,))
SCREENSHOT_SRC = re.compile(r"^data:image/webp;base64,[A-Za-z0-9+/]+=*$")
ASSET = re.compile(rf"!\[([^]]+)\]\({ASSETS_DIR}/([0-9a-f]{{16}}\.webp)\)")
PAGE_LINK = re.compile(
    r"\[Whole page (\d+)\]\(#(p-[a-z]+-\d+)\) \((\d+) findings?\)",
)


def _side(
    *segments: tuple[SegmentOp, str],
    source: SourceRef = SOURCE,
) -> Side:
    """Return a side of segments, at the source of a PDF block by default."""
    return Side(source, tuple(starmap(DiffSegment, segments)))


def _finding(
    severity: Severity = Severity.MAJOR,
    category: Category = Category.TEXT,
    *,
    ident: str = "f",
) -> Finding:
    """Return a finding whose both sides show a changed word."""
    return Finding(
        "text.typo",
        severity,
        category,
        "Typo",
        _side(("equal", "the "), ("delete", "word"), ("equal", " here")),
        _side(("equal", "the "), ("insert", "wrod"), ("equal", " here")),
        id=ident,
    )


def _pixels(size: tuple[float, float]) -> tuple[int, int]:
    """Return the pixels of a whole page in a screenshot."""
    return (
        math.ceil(size[0] * PAGE_SCALE),
        math.ceil(size[1] * PAGE_SCALE),
    )


def _result(*findings: Finding, fmt: str = "pdf") -> CompareResult:
    """Return a comparison of two documents of a format, with findings."""
    doc = replace(make_document(), path=Path("missing.pdf"), format=fmt)
    stats = CompareStats(ref_blocks=1, new_blocks=1, matched=1, moved=0)
    return CompareResult(doc, doc, list(findings), stats)


def test_a_changed_part_is_found_with_its_text() -> None:
    """A side with a changed part and text around it finds the part."""
    side = _side(("equal", "the "), ("delete", "word"), ("equal", " here"))
    assert shot_request(side) == ShotRequest(
        (BOX,),
        "find",
        "delete",
        Wanted("the ", "word", " here"),
    )


def test_a_whole_changed_side_tints_its_blocks() -> None:
    """A side whose one segment changed tints its blocks whole."""
    side = _side(("insert", "A new paragraph."))
    assert shot_request(side) == ShotRequest((BOX,), "whole", "insert")


@pytest.mark.parametrize(
    ("segments", "wanted"),
    [
        (
            (("equal", "the "), ("insert", ""), ("equal", " here")),
            Wanted("the ", "", " here"),
        ),
        (
            (("equal", "the core and"), ("insert", "")),
            Wanted("the core and", "", ""),
        ),
        (
            (("insert", ""), ("equal", "the shapes.")),
            Wanted("", "", "the shapes."),
        ),
    ],
    ids=["between", "end", "start"],
)
def test_an_empty_change_marks_its_place(
    segments: tuple[tuple[SegmentOp, str], ...],
    wanted: Wanted,
) -> None:
    """The side of removed text marks where the text was, at an edge too."""
    side = _side(*segments)
    assert shot_request(side) == ShotRequest((BOX,), "find", "insert", wanted)


def test_an_unchanged_side_tints_its_blocks_in_its_colors() -> None:
    """A side without change is tinted whole, in blue or in an audit yellow.

    In a comparison, such as a moved block or a list lost, the text of the
    block is kept, but not its place or its kind; in an audit, the finding
    points at the whole block.
    """
    side = _side(("equal", "A block that moved."))
    assert shot_request(side) == ShotRequest((BOX,), "whole", "block")
    assert shot_request(side, "mark") == ShotRequest((BOX,), "whole", "mark")


def test_a_marked_part_has_the_colors_of_a_mark() -> None:
    """The style of a side is that of its change, as a mark of an audit."""
    side = _side(("equal", "See "), ("mark", "??"), ("equal", " here."))
    request = shot_request(side)
    assert request is not None
    assert (request.mode, request.style) == ("find", "mark")


@pytest.mark.parametrize(
    "side",
    [
        None,
        _side(("delete", "text"), source=SourceRef("doc.pdf", page=1)),
        replace(_side(("delete", "")), image=None, source=SourceRef("a.html")),
    ],
    ids=["no-side", "no-boxes", "html"],
)
def test_sides_without_boxes_have_no_screenshot(side: Side | None) -> None:
    """A side needs the boxes of its blocks for a screenshot."""
    assert shot_request(side) is None


def test_sides_with_an_image_keep_their_thumbnail() -> None:
    """A side with an image gets no screenshot."""
    side = replace(_side(("delete", "")), image=object())
    assert shot_request(side) is None


def test_report_order_is_by_severity_then_category() -> None:
    """The findings are ordered by severity, then category, as reports do."""
    findings = [
        _finding(Severity.MINOR, Category.TEXT, ident="a"),
        _finding(Severity.CRITICAL, Category.LINK, ident="b"),
        _finding(Severity.MINOR, Category.CONTENT, ident="c"),
        _finding(Severity.CRITICAL, Category.CONTENT, ident="d"),
        _finding(Severity.MINOR, Category.TEXT, ident="e"),
    ]
    order = [finding.id for finding in report_order(findings)]
    assert order == ["d", "b", "c", "a", "e"]


def test_every_finding_has_screenshots_without_a_limit() -> None:
    """Without a limit, nothing tells of findings without screenshots."""
    findings = [_finding(ident=str(index)) for index in range(250)]
    assert Screenshots(_result(*findings)).note() is None


def test_the_limit_keeps_the_first_findings() -> None:
    """The limit of the configuration leaves the other findings out."""
    findings = [_finding(ident=str(index)) for index in range(5)]
    note = Screenshots(_result(*findings), limit=2).note()
    assert note == (
        "Only the first 2 findings have screenshots, as screenshots_max "
        "sets: 3 more have none."
    )


def test_each_listed_occurrence_of_a_group_counts() -> None:
    """The occurrences that a group lists count one each against the limit."""
    occurrences = tuple(_finding(ident=str(index)) for index in range(4))
    group = replace(_finding(ident="group"), occurrences=occurrences)
    note = Screenshots(_result(group), limit=1).note()
    assert note is not None
    assert "first 1 findings" in note
    assert "3 more" in note


def test_disabled_and_html_reports_have_no_screenshot() -> None:
    """Screenshots off, or documents in HTML, give none and no note."""
    finding = _finding()
    off = Screenshots(_result(finding), enabled=False, limit=1)
    html = Screenshots(_result(finding, _finding(), fmt="html"), limit=1)
    for screenshots in (off, html):
        assert screenshots.note() is None
        assert screenshots.get(finding, "ref") is None
        assert screenshots.page(finding, "ref") is None
        assert not screenshots.rows()


@pytest.mark.parametrize(
    ("matched", "ref", "new", "rows"),
    [
        (((1, 1), (2, 2), (3, 3)), {2}, {3}, [(2, 2), (3, 3)]),
        (((1, 1), (4, 4)), {2, 3}, {2}, [(2, 2), (3, None)]),
        (((5, 7),), {5}, set(), [(5, 7)]),
        ((), {3, 1}, set(), [(1, None), (3, None)]),
        (((2, 2),), {1}, {3}, [(1, None), (None, 3)]),
    ],
    ids=["matched", "between", "beside", "alone", "apart"],
)
def test_pages_are_laid_out_side_by_side(
    matched: tuple[tuple[int, int], ...],
    ref: set[int],
    new: set[int],
    rows: list[tuple[int | None, int | None]],
) -> None:
    """A matched page faces its match; the others face their peers between."""
    assert page_rows(matched, ref, new) == rows


def test_the_sides_of_a_finding_face_each_other_first() -> None:
    """A link of a finding outweighs all the text that links the pages."""
    links = {(2, 1): 500, (9, 2): 100}
    assert matched_pages(links, []) == ((2, 1), (9, 2))
    assert matched_pages(links, [(9, 1)]) == ((9, 1),)


@pytest.fixture(scope="module")
def pdf_result(tmp_path_factory: pytest.TempPathFactory) -> CompareResult:
    """Compare a PDF with another PDF of a changed number, with boxes."""
    root = tmp_path_factory.mktemp("shots")
    ref = build_reportlab_pdf(SAMPLE_EN, root / "ref.pdf")
    spec = replace_text(SAMPLE_EN, "100 requests", "10 requests")
    new = build_fpdf2_pdf(spec, root / "new.pdf")
    return run_compare(ref, new, Config(screenshots=True))


def _twice(result: CompareResult) -> tuple[CompareResult, Finding, Finding]:
    """Return a result with its finding and a copy of it, on its pages."""
    (finding,) = result.findings
    copy = replace(finding, id="copy")
    return replace(result, findings=[finding, copy]), finding, copy


def test_the_sides_of_a_pdf_finding_have_their_zooms(
    pdf_result: CompareResult,
) -> None:
    """Each side of a finding of two PDFs shows the place of its change."""
    (finding,) = pdf_result.findings
    screenshots = Screenshots(pdf_result)
    ref, new = screenshots.get(finding, "ref"), screenshots.get(finding, "new")
    assert ref is not None
    assert new is not None
    assert (ref.page, new.page) == (9, 1)
    assert ref.image.data[:4] == b"RIFF"
    # A few lines across their block, far less than a page.
    for zoom, size in ((ref, A4), (new, A5)):
        assert zoom.image.width >= ZOOM_MIN_WIDTH * ZOOM_SCALE
        assert zoom.image.width <= size[0] * ZOOM_SCALE
        assert zoom.image.height < size[1] * ZOOM_SCALE / 4
    assert screenshots.get(finding, "ref") is ref


def test_the_sides_of_a_pdf_finding_have_their_pages(
    pdf_result: CompareResult,
) -> None:
    """Each side of a finding of two PDFs is on its whole page."""
    (finding,) = pdf_result.findings
    screenshots = Screenshots(pdf_result)
    ref, new = (
        screenshots.page(finding, "ref"),
        screenshots.page(finding, "new"),
    )
    assert ref is not None
    assert new is not None
    assert (ref.title, ref.findings) == ("Reference, page 9", (finding,))
    assert (new.title, new.findings) == ("New, page 1", (finding,))
    # The whole pages: A4 for reportlab, A5 for fpdf2.
    assert (ref.image.width, ref.image.height) == _pixels(A4)
    assert (new.image.width, new.image.height) == _pixels(A5)
    assert screenshots.rows() == [PageRow(ref, new)]
    assert (ref.anchor, new.anchor) == ("p-ref-9", "p-new-1")


def test_a_page_is_shot_once_for_all_its_findings(
    pdf_result: CompareResult,
) -> None:
    """The findings on a page share it, listed in the order of the reports."""
    result, finding, copy = _twice(pdf_result)
    screenshots = Screenshots(result)
    (row,) = screenshots.rows()
    assert row.title == "Reference, page 9 \u00b7 New, page 1"
    for page in (row.ref, row.new):
        assert page is not None
        assert page.findings == (finding, copy)
        assert screenshots.page(finding, page.column.which) is page
        assert screenshots.page(copy, page.column.which) is page


def test_the_pages_have_the_findings_of_the_limit_only(
    pdf_result: CompareResult,
) -> None:
    """A finding without screenshots is on no page."""
    result, finding, copy = _twice(pdf_result)
    screenshots = Screenshots(result, limit=1)
    (row,) = screenshots.rows()
    assert row.ref is not None
    assert row.new is not None
    assert (row.ref.findings, row.new.findings) == ((finding,), (finding,))
    assert screenshots.page(copy, "ref") is None
    assert screenshots.get(copy, "ref") is None


def test_the_html_report_shows_the_screenshots(
    pdf_result: CompareResult,
    tmp_path: Path,
) -> None:
    """The HTML report shows the zoom of each side, then a link to its page."""
    path = render_html(
        pdf_result,
        tmp_path / "report.html",
        None,
        Screenshots(pdf_result),
    )
    tree = lxml.html.fromstring(path.read_bytes())
    shots = tree.xpath("//div[contains(@class, 'side')]/img[@class='shot']")
    assert [shot.get("alt") for shot in shots] == [
        "Reference, page 9",
        "New, page 1",
    ]
    assert all(SCREENSHOT_SRC.match(shot.get("src")) for shot in shots)
    assert all(shot.getprevious().get("class") == "text" for shot in shots)
    links = [shot.getnext() for shot in shots]
    assert [link.text_content() for link in links] == [
        "Whole page 9 (1 finding)",
        "Whole page 1 (1 finding)",
    ]
    assert [link.find("a").get("href") for link in links] == [
        "#p-ref-9",
        "#p-new-1",
    ]
    (row,) = tree.xpath("//section[@id='pages']/article[@class='page']")
    pages = row.xpath("./div[contains(@class, 'sides')]/div[@id]")
    assert [page.get("id") for page in pages] == ["p-ref-9", "p-new-1"]
    assert [page.findtext("h3") for page in pages] == [
        "Reference, page 9",
        "New, page 1",
    ]
    images = tree.xpath("//img[@class='page-shot']")
    assert [image.get("alt") for image in images] == [
        "Reference, page 9, with its findings",
        "New, page 1, with its findings",
    ]
    assert all(SCREENSHOT_SRC.match(image.get("src")) for image in images)
    assert len(tree.xpath("//@src")) == len(shots) + len(images)
    (article,) = tree.xpath("//article[@class='finding']")
    for page in pages:
        (back,) = page.xpath(".//ul[@class='on-page']/li/a")
        assert back.get("href") == f"#{article.get('id')}"
    (header,) = tree.xpath("//header//a[@href='#pages']")
    assert header.text == "pages"
    policy = tree.xpath("//meta[@http-equiv='Content-Security-Policy']")[0]
    assert "img-src data:;" in policy.get("content")


def test_the_html_pages_link_to_the_occurrences_of_a_group(
    pdf_result: CompareResult,
    tmp_path: Path,
) -> None:
    """A page links to each occurrence on it, which a group lists."""
    (finding,) = pdf_result.findings
    occurrences = (replace(finding, id="one"), replace(finding, id="two"))
    group = replace(finding, id="group", occurrences=occurrences)
    result = replace(pdf_result, findings=[group])
    path = render_html(
        result,
        tmp_path / "report.html",
        None,
        Screenshots(result),
    )
    tree = lxml.html.fromstring(path.read_bytes())
    listed = tree.xpath("//details[@class='occurrences']/ol/li/@id")
    assert listed == ["f-one", "f-two"]
    pages = tree.xpath("//article[@class='page']//div[@id]")
    assert len(pages) == 2
    for page in pages:
        links = page.xpath(".//ul[@class='on-page']/li/a/@href")
        assert links == ["#f-one", "#f-two"]


def test_the_markdown_report_writes_the_screenshots(
    pdf_result: CompareResult,
    tmp_path: Path,
) -> None:
    """The Markdown report shows the zooms and the pages, next to it."""
    result, _, _ = _twice(pdf_result)
    path = render_markdown(
        result,
        tmp_path / "report.md",
        None,
        Screenshots(result),
    )
    text = path.read_text(encoding="utf-8")
    found = ASSET.findall(text)
    zooms = ["Reference, page 9", "New, page 1"]
    assert [label for label, _ in found] == [
        *zooms,
        *zooms,
        "Reference, page 9, with its findings",
        "New, page 1, with its findings",
    ]
    for _, name in found:
        assert (tmp_path / ASSETS_DIR / name).read_bytes()[:4] == b"RIFF"
    assert "<br>![Reference, page 9]" in text
    links = PAGE_LINK.findall(text)
    assert links == [("9", "p-ref-9", "2"), ("1", "p-new-1", "2")] * 2
    pages = text[text.index("\n## Pages (1)\n") :]
    assert "\n### Reference, page 9 \u00b7 New, page 1\n" in pages
    assert '\n<a name="p-ref-9"></a><a name="p-new-1"></a>\n' in pages
    assert "\n| Reference | New |\n|---|---|\n| ![Reference, page 9" in pages
    assert "<br>Number changed (`text.number_changed`), ID: `copy`" in pages


def test_reports_without_screenshots_are_unchanged(
    pdf_result: CompareResult,
    tmp_path: Path,
) -> None:
    """Screenshots off, the reports show none, as before."""
    html = render_html(pdf_result, tmp_path / "report.html")
    markdown = render_markdown(pdf_result, tmp_path / "report.md")
    tree = lxml.html.fromstring(html.read_bytes())
    assert not tree.xpath("//img")
    assert not tree.xpath("//*[@id='pages']")
    text = markdown.read_text(encoding="utf-8")
    assert not ASSET.findall(text)
    assert "## Pages" not in text
    assert not (tmp_path / ASSETS_DIR).exists()


def test_the_reports_tell_when_the_limit_left_screenshots_out(
    pdf_result: CompareResult,
    tmp_path: Path,
) -> None:
    """The note of the limit is in both reports."""
    result, _, _ = _twice(pdf_result)
    screenshots = Screenshots(result, limit=1)
    html = render_html(result, tmp_path / "report.html", None, screenshots)
    markdown = render_markdown(result, tmp_path / "r.md", None, screenshots)
    note = "Only the first 1 findings have screenshots"
    tree = lxml.html.fromstring(html.read_bytes())
    assert tree.xpath("//p[@class='shots']")[0].text.startswith(note)
    assert note in markdown.read_text(encoding="utf-8")
    assert len(tree.xpath("//img[@class='shot']")) == 2
    assert len(tree.xpath("//img[@class='page-shot']")) == 2


def test_a_code_finding_shows_its_screenshots_before_its_diff(
    tmp_path: Path,
) -> None:
    """The screenshots of a change of code come before its diff."""
    code = ["def main():", "    return compute(1, 2)", "", "main()"]
    edited = [line.replace("compute(1, 2)", "compute(1, 3)") for line in code]
    ref = build_listing_pdf(code, tmp_path / "ref.pdf", numbers=False)
    new = build_listing_pdf(edited, tmp_path / "new.pdf", numbers=False)
    result = run_compare(ref, new, Config(screenshots=True))
    path = render_markdown(
        result,
        tmp_path / "report.md",
        None,
        Screenshots(result),
    )
    text = path.read_text(encoding="utf-8")
    shots = text.index("![Reference, page 1]")
    assert text.index("[Whole page 1](#p-ref-1)") > shots
    assert text.index("![New, page 1]") > shots
    assert text.index("```diff") > text.index("[Whole page 1](#p-new-1)")


def test_an_audit_of_a_pdf_shows_its_marks(tmp_path: Path) -> None:
    """An audit shows the place of its findings, and their page, marked."""
    path = build_lines_pdf(
        [["See Section ?? for the limits."]],
        tmp_path / "doc.pdf",
    )
    cfg = Config(screenshots=True, checks=("structure",))
    result: AuditResult = run_audit(path, cfg)
    (finding,) = [
        f
        for f in result.findings
        if f.rule == "structure.unresolved_reference"
    ]
    screenshots = Screenshots(result)
    shot = screenshots.get(finding, "ref")
    assert shot is not None
    assert shot.page == 1
    page = screenshots.page(finding, "ref")
    assert isinstance(page, PageShot)
    assert (page.page, page.findings) == (1, (finding,))


def test_a_page_without_findings_is_shown_beside_its_match(
    tmp_path: Path,
) -> None:
    """The new page of a removed paragraph is shown, without marks."""
    first = ["The first paragraph stays as it is."]
    second = ["The second paragraph is removed from the new document."]
    third = ["The third paragraph stays as well."]
    ref = build_lines_pdf([first, second, third], tmp_path / "ref.pdf")
    new = build_lines_pdf([first, third], tmp_path / "new.pdf")
    result = run_compare(ref, new, Config(screenshots=True))
    (row,) = Screenshots(result).rows()
    assert row.ref is not None
    assert row.new is not None
    assert (row.ref.page, row.new.page) == (1, 1)
    assert row.ref.findings
    assert row.new.findings == ()
    assert row.new.image.data[:4] == b"RIFF"
    path = render_html(
        result,
        tmp_path / "report.html",
        None,
        Screenshots(result),
    )
    tree = lxml.html.fromstring(path.read_bytes())
    (beside,) = tree.xpath("//div[@id='p-new-1']")
    assert beside.findtext("p") == "No findings on this page."
    assert beside.find("img").get("alt") == "New, page 1"


def test_pages_that_moved_face_their_matches(tmp_path: Path) -> None:
    """A page of inserted text is alone; the others face their matches.

    The new document starts with a paragraph of its own, which pushes the
    long paragraph of the reference: the word that changed, on the second
    page of the reference, is on the third page of the new document, and
    the start of the long paragraph, on the first page of the reference, is
    mostly on the second.
    """
    names = ["amber", "birch", "cedar", "dune", "ember"]
    names += ["fjord", "grove", "heath", "inlet", "jetty"]
    lines = [f"the {one} by the {two} bank" for one in names for two in names]
    ref = build_lines_pdf(
        [[*lines[:70], "The word to change is here.", *lines[71:]]],
        tmp_path / "ref.pdf",
    )
    new = build_lines_pdf(
        [
            [f"an inserted {one} line" for one in names] * 5,
            [*lines[:70], "The term to change is here.", *lines[71:]],
        ],
        tmp_path / "new.pdf",
    )
    result = run_compare(ref, new, Config(screenshots=True))
    rows = Screenshots(result).rows()
    assert [
        tuple(None if page is None else page.page for page in pages)
        for pages in ((row.ref, row.new) for row in rows)
    ] == [(None, 1), (2, 3)]
    assert rows[1].ref is not None
    assert rows[1].new is not None
    assert rows[1].ref.findings == rows[1].new.findings
