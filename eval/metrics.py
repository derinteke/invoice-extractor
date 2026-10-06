"""Field- and item-level scoring of an extraction against labelled ground truth."""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

from invoice_extractor.normalize import amounts_equal, norm_id, norm_text
from invoice_extractor.schema import NUMERIC_FIELDS, Document, LineItem, purchased

ID_FIELDS = {"invoice_number", "vendor_tax_id", "customer_tax_id", "iban"}
NAME_FIELDS = {"vendor_name", "customer_name"}
ITEM_MATCH_THRESHOLD = 0.7


def field_correct(name: str, pred, truth) -> bool:
    if name in NUMERIC_FIELDS:
        return amounts_equal(pred, truth)
    if name in ID_FIELDS:
        return bool(norm_id(pred)) and norm_id(pred) == norm_id(truth)
    if name in NAME_FIELDS:
        # Labels may include the address after the name ("Clay-Koch 2467 John Knolls ...").
        # Keep hyphens/dots so a truncated "Clay" doesn't match "Clay-Koch".
        p = " ".join(str(pred or "").casefold().split()).rstrip(",;")
        t = " ".join(str(truth or "").casefold().split())
        return len(p) >= 3 and (p == t or t.startswith(p + " ") or t.startswith(p + ","))
    if name == "date":
        return pred == truth
    return norm_text(pred) == norm_text(truth)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def match_items(pred: list[LineItem], truth: list[LineItem]) -> list[tuple[LineItem, LineItem]]:
    """Greedy one-to-one pairing by description similarity."""
    candidates = sorted(
        ((_similarity(p.description, t.description), i, j)
         for i, p in enumerate(pred) for j, t in enumerate(truth)),
        reverse=True,
    )
    used_p, used_t, pairs = set(), set(), []
    for score, i, j in candidates:
        if score < ITEM_MATCH_THRESHOLD:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        pairs.append((pred[i], truth[j]))
    return pairs


def item_correct(pred: LineItem, truth: LineItem) -> bool:
    """A matched line is correct when every labelled number agrees."""
    for attr in ("quantity", "unit_price", "total"):
        t = getattr(truth, attr)
        if t is not None and not amounts_equal(getattr(pred, attr), t):
            return False
    return True


@dataclass
class DocScore:
    fields: dict[str, bool] = field(default_factory=dict)
    items_pred: int = 0
    items_truth: int = 0
    items_matched: int = 0
    items_correct: int = 0

    @property
    def exact(self) -> bool:
        """Every labelled field right and every line item found and right."""
        return all(self.fields.values()) and self.items_correct == self.items_truth == self.items_pred


def score_document(pred: Document | None, truth: dict, truth_items: list[LineItem]) -> DocScore:
    if pred is None:  # failed extraction: everything counts as wrong
        return DocScore({k: False for k in truth}, 0, len(truth_items), 0, 0)
    pred_items = purchased(pred.line_items)
    score = DocScore(
        fields={k: field_correct(k, getattr(pred, k), v) for k, v in truth.items()},
        items_pred=len(pred_items),
        items_truth=len(truth_items),
    )
    pairs = match_items(pred_items, truth_items)
    score.items_matched = len(pairs)
    score.items_correct = sum(item_correct(p, t) for p, t in pairs)
    return score


def aggregate(scores: list[DocScore]) -> dict:
    per_field: dict[str, list[bool]] = {}
    for s in scores:
        for k, ok in s.fields.items():
            per_field.setdefault(k, []).append(ok)
    pred = sum(s.items_pred for s in scores)
    truth = sum(s.items_truth for s in scores)
    correct = sum(s.items_correct for s in scores)
    precision = correct / pred if pred else 0.0
    recall = correct / truth if truth else 0.0
    all_fields = [ok for oks in per_field.values() for ok in oks]
    return {
        "documents": len(scores),
        "field_accuracy": {k: sum(v) / len(v) for k, v in sorted(per_field.items())},
        "field_accuracy_micro": sum(all_fields) / len(all_fields) if all_fields else None,
        "item_precision": precision,
        "item_recall": recall,
        "item_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "doc_exact": sum(s.exact for s in scores) / len(scores) if scores else 0.0,
    }
