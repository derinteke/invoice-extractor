"""Deterministic consistency checks on an extraction.

The LLM reads the document; code checks that what it read adds up. A document that fails any
check is routed to human review instead of being auto-accepted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .normalize import amounts_equal
from .schema import Document, purchased


@dataclass(frozen=True)
class Issue:
    code: str
    message: str


def _iban_valid(iban: str) -> bool:
    s = re.sub(r"\s", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(digits) % 97 == 1


def _close(a: float, b: float) -> bool:
    return amounts_equal(a, b, rel=0.01)


def check(doc: Document) -> list[Issue]:
    issues: list[Issue] = []

    if doc.total is None:
        issues.append(Issue("missing_total", "No total amount was found."))

    items = purchased(doc.line_items)
    for i, item in enumerate(items, 1):
        if None not in (item.quantity, item.unit_price, item.total):
            if not amounts_equal(item.quantity * item.unit_price, item.total, rel=0.01):
                issues.append(Issue(
                    "line_math",
                    f"Line {i}: {item.quantity} x {item.unit_price} != {item.total}.",
                ))

    line_totals = [it.total for it in items if it.total is not None]
    if line_totals and len(line_totals) == len(items):
        gross = round(sum(line_totals), 2)
        net = round(gross + sum(it.discount or 0 for it in items), 2)
        target = doc.subtotal
        if target is None and doc.total is not None:
            target = doc.total - (doc.tax or 0) - (doc.service_charge or 0) - (doc.discount or 0)
        # Receipts print the subtotal either before or after line discounts.
        if target is not None and not any(_close(x, target) for x in (gross, net)):
            issues.append(Issue(
                "lines_vs_subtotal",
                f"Line items add up to {net:g}, but the subtotal is {target:g}.",
            ))

    if doc.subtotal is not None and doc.total is not None:
        extras = (doc.service_charge or 0) + (doc.discount or 0)
        # Accepted printings: tax added on top, or prices already tax-inclusive.
        candidates = [doc.subtotal + extras + (doc.tax or 0), doc.subtotal + extras]
        if not any(_close(c, doc.total) for c in candidates):
            parts = " + ".join(
                f"{name} {value:g}" for name, value in [
                    ("subtotal", doc.subtotal), ("discount", doc.discount),
                    ("service", doc.service_charge), ("tax", doc.tax)] if value is not None)
            issues.append(Issue("totals_math", f"{parts} != total {doc.total:g}."))

    if doc.date:
        try:
            d = date.fromisoformat(doc.date)
            if d > date.today():
                issues.append(Issue("future_date", f"Date {doc.date} is in the future."))
        except ValueError:
            issues.append(Issue("bad_date", f"Date '{doc.date}' is not YYYY-MM-DD."))

    if doc.iban and not _iban_valid(doc.iban):
        issues.append(Issue("bad_iban", f"IBAN {doc.iban} fails the checksum."))

    return issues
