# Copyright 2026 Christophe Le Douarec
"""Standalone HTML report of a comparison or of an audit.

The report is one file that opens from the disk: its style sheet and its
script are inline, and a content security policy lets them run, but nothing
else. The texts of the documents are escaped, and no URL of a document ever
becomes the target of a link or the source of a resource. The only images are
the WebP thumbnails and screenshots that Pillow encodes again, in data URLs;
an SVG is never written. Each page that the screenshots show is in the report
once, in its Pages section, beside the page of the other document that
matches it: the findings on the page link to it, and it links to them.
"""

import base64
import hashlib
import importlib.resources
from collections import Counter
from typing import TYPE_CHECKING

from documentation_checker.model.findings import Severity
from documentation_checker.report.context import (
    CATEGORY_TITLES,
    SEVERITY_TITLES,
    TEMPLATES,
    ReportContext,
    Thumbnails,
    counted,
    is_code,
    side_of,
)
from documentation_checker.report.screenshots import Screenshots

if TYPE_CHECKING:
    from pathlib import Path

    from documentation_checker.audit.engine import AuditResult
    from documentation_checker.compare.engine import CompareResult
    from documentation_checker.model.findings import Finding


def _hash_source(text: str) -> str:
    """Return the source expression that allows one inline text."""
    digest = hashlib.sha256(text.encode()).digest()
    return f"'sha256-{base64.b64encode(digest).decode()}'"


def content_security_policy(css: str, script: str) -> str:
    """Return the content security policy of a report.

    Args:
        css: Inline style sheet, exactly as its ``style`` element holds it.
        script: Inline script, exactly as its ``script`` element holds it.

    Returns:
        A policy that allows the style sheet and the script by their SHA-256
        hashes, and images in data URLs, but no other resource, no base URL
        and no form submission.
    """
    return (
        "default-src 'none'; img-src data:; "
        f"style-src {_hash_source(css)}; "
        f"script-src {_hash_source(script)}; "
        "base-uri 'none'; form-action 'none'"
    )


def _asset(name: str) -> str:
    """Return the text of a style sheet or a script of the report."""
    resource = importlib.resources.files(
        "documentation_checker.report",
    ).joinpath(
        f"templates/{name}",
    )
    return resource.read_text(encoding="utf-8")


class _DomIds:
    """Identifiers of the DOM, unique even when finding identifiers repeat."""

    def __init__(self) -> None:
        """Start with no identifier given out."""
        self._given: Counter[str] = Counter()
        self._first: dict[int, str] = {}

    def __call__(self, finding: Finding) -> str:
        """Return the next DOM identifier of a finding.

        Args:
            finding: The finding, or an occurrence of a group, whose
                identifier identical findings share.

        Returns:
            ``f-{id}`` the first time, then ``f-{id}-2``, ``f-{id}-3``...
        """
        self._given[finding.id] += 1
        count = self._given[finding.id]
        given = f"f-{finding.id}" if count == 1 else f"f-{finding.id}-{count}"
        self._first.setdefault(id(finding), given)
        return given

    def of(self, finding: Finding) -> str | None:
        """Return the DOM identifier first given to a finding, if any.

        Args:
            finding: The finding, or an occurrence of a group.

        Returns:
            The identifier, ``None`` if the report has not shown it yet.
        """
        return self._first.get(id(finding))


def render_html(
    result: CompareResult | AuditResult,
    path: Path,
    thumbnails: Thumbnails | None = None,
    screenshots: Screenshots | None = None,
) -> Path:
    """Write the standalone HTML report of a comparison or of an audit.

    Args:
        result: Result of a comparison or of an audit.
        path: Report file to write.
        thumbnails: Thumbnails of the sides of the findings, shared with the
            other report; ``None`` for thumbnails of this report only.
        screenshots: Screenshots of the sides of the findings, shared with
            the other report; ``None`` for no screenshots.

    Returns:
        The written file.
    """
    if thumbnails is None:
        with Thumbnails(result) as own:
            return render_html(result, path, own, screenshots)
    if screenshots is None:
        screenshots = Screenshots(result, enabled=False)
    context = ReportContext(result, path.parent)
    css = _asset("report.css")
    script = _asset("report.js")
    severities = sorted(Severity, reverse=True)
    dom_ids = _DomIds()
    # The report is written as the template makes it: the data URLs of the
    # images can make it large.
    chunks = TEMPLATES.get_template("report.html.j2").generate(
        policy=content_security_policy(css, script),
        css=css,
        script=script,
        command=context.command,
        columns=context.columns,
        inputs=context.inputs(),
        **context.headline(),
        severities=severities,
        counts=[
            (severity, len(context.findings_of(severity)))
            for severity in severities
        ],
        categories=context.categories(),
        rules=sorted({finding.rule for finding in result.findings}),
        summary=context.summary(),
        normalization=context.normalization(),
        chrome=context.chrome(),
        warnings=context.warnings(),
        sections=[
            (severity, [f for _, findings in groups for f in findings])
            for severity, groups in context.sections()
        ],
        total=len(result.findings),
        category_titles=CATEGORY_TITLES,
        severity_titles=SEVERITY_TITLES,
        source=context.source,
        side_of=side_of,
        shown=context.shown,
        hidden=context.hidden,
        is_code=is_code,
        thumbnail=thumbnails.get,
        screenshot=screenshots.get,
        page_shot=screenshots.page,
        screenshot_note=screenshots.note(),
        rows=screenshots.rows(),
        counted=counted,
        dom_id=dom_ids,
        target=dom_ids.of,
    )
    with path.open("w", encoding="utf-8", newline="\n") as report:
        report.writelines(chunks)
    return path
