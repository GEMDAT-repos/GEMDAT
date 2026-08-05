"""This module contains the exceptions raised by gemdat."""

from __future__ import annotations


class NotSupportedError(NotImplementedError):
    """Raised when the input uses a feature that gemdat does not (yet) support.

    Subclasses [NotImplementedError][] so that it reads as a gap in
    gemdat rather than a mistake by the caller.
    """
