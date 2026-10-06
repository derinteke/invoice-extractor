"""Fields that code derives after the LLM has read the document."""
from __future__ import annotations

import os

from .normalize import resolve_date
from .schema import Document


def finalize(doc: Document, date_order: str | None = None) -> Document:
    """Resolve the ISO date from the printed one using the source's date convention."""
    order = date_order or os.getenv("DATE_ORDER", "DMY")
    return doc.model_copy(update={"date": resolve_date(doc.date_printed, order)})
