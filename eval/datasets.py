"""Public labelled datasets mapped onto the extraction schema.

- CORD v2 (naver-clova-ix/cord-v2, CC BY 4.0): 100 Indonesian shop receipts, IDR amounts
  printed with '.' as thousands separator. Labels: line items, subtotal, tax, total.
- Invoices (katanaml-org/invoices-donut-data-v1): 26 synthetic English invoices.
  Labels: number, date, seller/client (+ tax ids), IBAN, line items, totals.

Only fields that a dataset actually labels are scored for it.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator
from dataclasses import dataclass, field

from invoice_extractor.loaders import Payload, image_to_jpeg
from invoice_extractor.normalize import parse_amount, parse_date
from invoice_extractor.schema import LineItem

DATASETS = {
    "cord": "naver-clova-ix/cord-v2",
    "invoices": "katanaml-org/invoices-donut-data-v1",
}
# How ambiguous dates like 10/01/2014 are printed in each source (see normalize.resolve_date).
DATE_ORDER = {"cord": "DMY", "invoices": "MDY"}
LABEL_FIXES = json.loads((Path(__file__).parent / "label_fixes.json").read_text(encoding="utf-8"))


@dataclass
class Sample:
    id: str
    dataset: str
    payload: Payload
    truth: dict = field(default_factory=dict)        # scalar field -> expected value
    truth_items: list[LineItem] = field(default_factory=list)


def _first(value):
    """CORD stores repeated keys as lists; keep the first value."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def map_cord(gt: dict) -> tuple[dict, list[LineItem]]:
    truth: dict = {}
    sub_total = gt.get("sub_total") or {}
    total = gt.get("total") or {}
    if isinstance(sub_total, list):
        sub_total = sub_total[0]
    if isinstance(total, list):
        total = total[0]
    for key, source in [("subtotal", sub_total.get("subtotal_price")),
                        ("discount", sub_total.get("discount_price")),
                        ("service_charge", sub_total.get("service_price")),
                        ("tax", sub_total.get("tax_price")),
                        ("total", total.get("total_price"))]:
        amount = parse_amount(_first(source))
        if amount is not None and not (key == "discount" and amount == 0):
            truth[key] = -abs(amount) if key == "discount" else amount

    def item(m: dict) -> LineItem | None:
        name = _first(m.get("nm"))
        if not name:
            return None
        discount = parse_amount(_first(m.get("discountprice")))
        return LineItem(
            description=str(name),
            quantity=parse_amount(_first(m.get("cnt"))),
            unit_price=parse_amount(_first(m.get("unitprice"))),
            total=parse_amount(_first(m.get("price"))),
            discount=-abs(discount) if discount else None,
        )

    items = []
    for m in _as_list(gt.get("menu")):
        if (main := item(m)) is not None:
            items.append(main)
        # Priced add-ons are purchases; zero-priced options (size, "- less ice") are not.
        for sub in _as_list(m.get("sub")):
            if isinstance(sub, dict) and (added := item(sub)) is not None and added.total:
                items.append(added)
    return truth, items


def map_invoice(gt: dict) -> tuple[dict, list[LineItem]]:
    header, summary = gt.get("header", {}), gt.get("summary", {})
    truth = {
        "invoice_number": header.get("invoice_no"),
        "date": parse_date(header.get("invoice_date"), ["%m/%d/%Y"]),
        "vendor_name": header.get("seller"),          # name + address; scored as a prefix
        "customer_name": header.get("client"),
        "vendor_tax_id": header.get("seller_tax_id"),
        "customer_tax_id": header.get("client_tax_id"),
        "iban": header.get("iban"),
        "subtotal": parse_amount(summary.get("total_net_worth")),
        "tax": parse_amount(summary.get("total_vat")),
        "total": parse_amount(summary.get("total_gross_worth")),
    }
    items = [
        LineItem(
            description=it.get("item_desc", ""),
            quantity=parse_amount(it.get("item_qty")),
            unit_price=parse_amount(it.get("item_net_price")),
            total=parse_amount(it.get("item_net_worth")),
        )
        for it in gt.get("items", [])
    ]
    return {k: v for k, v in truth.items() if v not in (None, "")}, items


MAPPERS = {"cord": map_cord, "invoices": map_invoice}


def load_samples(name: str, split: str = "test", limit: int | None = None,
                 fix_labels: bool = True) -> Iterator[Sample]:
    from datasets import load_dataset

    ds = load_dataset(DATASETS[name], split=split)
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        row = ds[i]
        truth, items = MAPPERS[name](json.loads(row["ground_truth"])["gt_parse"])
        sample_id = f"{name}-{split}-{i:03d}"
        if fix_labels:
            for key, fix in LABEL_FIXES.get(sample_id, {}).items():
                truth[key] = fix["value"]
        yield Sample(
            id=sample_id,
            dataset=name,
            payload=Payload(image_to_jpeg(row["image"]), "image/jpeg"),
            truth=truth,
            truth_items=items,
        )
