# Copyright 2026 Christophe Le Douarec
"""Errors raised by Documentation Checker."""


class DocumentationCheckerError(Exception):
    """Base class of the errors reported to the user."""


class ConfigError(DocumentationCheckerError):
    """Invalid configuration file or option value."""


class InputError(DocumentationCheckerError):
    """Unreadable or unsupported input document."""


class DictionaryError(DocumentationCheckerError):
    """A bundled dictionary that cannot be loaded."""
