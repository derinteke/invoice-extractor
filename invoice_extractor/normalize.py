"""Parsing helpers for amounts, dates and identifiers printed in many locales."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime

_AMOUNT_CHARS = re.compile(r"[^\d.,\-]")


def parse_amount(value) -> float | None:
    """Parse '$ 1.234,56', '60.000' (IDR thousands), '1,234.56', '2,00' ... into a float.

    A single separator followed by exactly three digits is read as a thousands separator,
    which is how receipts in IDR/TRY/EUR locales print whole amounts.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = _AMOUNT_CHARS.sub("", str(value))
    negative = s.startswith("-")
    s = s.strip("-")
    if not re.search(r"\d", s):
        return None
    s = s.strip(".,")
    last_dot, last_comma = s.rfind("."), s.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        # Both present: the right-most one is the decimal separator.
        dec = "." if last_dot > last_comma else ","
        thousands = "," if dec == "." else "."
        s = s.replace(thousands, "").replace(dec, ".")
    elif last_dot >= 0 or last_comma >= 0:
        sep = "." if last_dot >= 0 else ","
        parts = s.split(sep)
        if len(parts) > 2 or len(parts[-1]) == 3:
            s = s.replace(sep, "")  # thousands separator(s)
        else:
            s = s.replace(sep, ".")
    try:
        number = float(s)
    except ValueError:
        return None
    return -number if negative else number


def amounts_equal(a: float | None, b: float | None, rel: float = 0.0, abs_tol: float = 0.011) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(abs_tol, rel * abs(b))


_DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%d.%m.%y", "%m/%d/%y",
]


def parse_date(value: str | None, formats: list[str] | None = None) -> str | None:
    """Return an ISO date (YYYY-MM-DD) or None. Order of `formats` decides ambiguous dates."""
    if not value:
        return None
    s = str(value).strip()
    for fmt in formats or _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def norm_text(value: str | None) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace."""
    if not value:
        return ""
    s = unicodedata.normalize("NFKD", str(value).replace("ı", "i"))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def norm_id(value: str | None) -> str:
    """Identifiers (tax ids, IBANs, invoice numbers) compared on letters and digits only."""
    return re.sub(r"[^0-9A-Za-z]", "", str(value or "")).upper()


def resolve_date(printed: str | None, order: str = "DMY") -> str | None:
    """ISO date from a printed date. `order` (DMY or MDY) only decides truly ambiguous dates.

    "18/03/2017" and "03/18/2017" are unambiguous; "10/01/2014" is read with `order`,
    which is a property of where the documents come from, not something to guess per page.
    """
    if not printed:
        return None
    s = printed.strip()
    iso = parse_date(s, ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"])
    if iso:
        return iso
    parts = re.findall(r"\d+", s)
    if len(parts) == 3 and not re.search(r"[A-Za-z]{3}", s):
        a, b, y = (int(x) for x in parts)
        if len(parts[0]) == 4:
            a, b, y = b, y, a
        if y < 100:
            y += 2000
        if a > 12 or (b <= 12 and order.upper() == "DMY"):
            day, month = a, b
        else:
            day, month = b, a
        try:
            return datetime(y, month, day).date().isoformat()
        except ValueError:
            return None
    return parse_date(s)
