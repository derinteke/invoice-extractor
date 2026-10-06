import json
from types import SimpleNamespace

import pytest

from eval.datasets import map_cord, map_invoice
from eval.metrics import aggregate, field_correct, score_document
from invoice_extractor.extractors import ExtractionError, GeminiExtractor, _parse
from invoice_extractor.loaders import Payload
from invoice_extractor.normalize import parse_amount, parse_date
from invoice_extractor.schema import Document, LineItem, gemini_json_schema
from invoice_extractor.validation import check


@pytest.mark.parametrize("raw, expected", [
    ("60.000", 60000), ("-60.000", -60000), ("5.455", 5455), ("1.234.567", 1234567),
    ("$ 889,20", 889.2), ("2,00", 2), ("1,234.56", 1234.56), ("1.234,56", 1234.56),
    ("x2", 2), ("2.00", 2), ("Rp 15.000,-", 15000), ("", None), ("abc", None), (12, 12.0),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


def test_parse_date_respects_format_order():
    assert parse_date("09/07/2015", ["%m/%d/%Y"]) == "2015-09-07"
    assert parse_date("07.09.2015") == "2015-09-07"
    assert parse_date("not a date") is None


def test_schema_has_no_refs():
    assert "$ref" not in json.dumps(gemini_json_schema())


def _doc(**kw):
    return Document(doc_type="invoice", **kw)


def test_validation_clean_document():
    doc = _doc(line_items=[LineItem(description="a", quantity=2, unit_price=10, total=20)],
               subtotal=20, tax=2, total=22, date="2015-09-07", iban="GB82WEST12345698765432")
    assert check(doc) == []


def test_validation_flags_math_and_iban():
    doc = _doc(line_items=[LineItem(description="a", quantity=2, unit_price=10, total=25)],
               subtotal=20, tax=2, total=30, iban="GB00WEST12345698765432")
    codes = {i.code for i in check(doc)}
    assert codes == {"line_math", "lines_vs_subtotal", "totals_math", "bad_iban"}


def test_validation_tax_inclusive_receipt_is_ok():
    # Receipts often print a subtotal that already includes tax.
    assert check(_doc(subtotal=75000, tax=6818, total=75000)) == []


def test_map_cord_handles_lists_and_missing_fields():
    gt = {"menu": [{"nm": "Tea", "cnt": "2", "price": "20.000"}, {"nm": ["Cake", "x"], "price": "15.000"}],
          "total": {"total_price": "35.000"}}
    truth, items = map_cord(gt)
    assert truth == {"total": 35000}
    assert [(i.description, i.quantity, i.total) for i in items] == [("Tea", 2, 20000), ("Cake", None, 15000)]


def test_map_invoice():
    gt = {"header": {"invoice_no": "1", "invoice_date": "09/18/2015", "seller": "Acme 1 Road"},
          "items": [{"item_desc": "x", "item_qty": "2,00", "item_net_price": "1,50", "item_net_worth": "3,00"}],
          "summary": {"total_gross_worth": "$ 3,30"}}
    truth, items = map_invoice(gt)
    assert truth == {"invoice_number": "1", "date": "2015-09-18", "vendor_name": "Acme 1 Road", "total": 3.3}
    assert items[0].total == 3.0


def test_field_comparators():
    assert field_correct("vendor_name", "Clay-Koch", "Clay-Koch 2467 John Knolls")
    assert not field_correct("vendor_name", "Clay", "Clay-Koch 2467")  # partial name is wrong
    assert field_correct("vendor_tax_id", "915 72 3800", "915-72-3800")
    assert field_correct("total", 4391.1, 4391.10)
    assert not field_correct("total", 4391.0, 4391.1)
    assert not field_correct("iban", None, "GB82")


def test_score_document_and_aggregate():
    truth = {"total": 22.0, "invoice_number": "A1"}
    truth_items = [LineItem(description="Blue widget", quantity=2, total=20),
                   LineItem(description="Red gadget", quantity=1, total=2)]
    pred = _doc(total=22, invoice_number="B1", line_items=[
        LineItem(description="Blue widget", quantity=2, total=20),
        LineItem(description="Red gadgets", quantity=1, total=3),     # matched, wrong amount
        LineItem(description="Delivery", total=0),                    # extra line
    ])
    s = score_document(pred, truth, truth_items)
    assert s.fields == {"total": True, "invoice_number": False}
    assert (s.items_pred, s.items_truth, s.items_matched, s.items_correct) == (3, 2, 2, 1)
    assert not s.exact
    agg = aggregate([s, score_document(None, truth, truth_items)])
    assert agg["field_accuracy"]["total"] == 0.5
    assert agg["item_recall"] == 0.25


def test_parse_strips_code_fence_and_rejects_bad_output():
    assert _parse('```json\n{"doc_type": "receipt", "total": 5}\n```').total == 5
    with pytest.raises(ExtractionError):
        _parse('{"doc_type": "letter"}')


class FakeModels:
    def __init__(self, text):
        self.text, self.calls = text, []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        return SimpleNamespace(text=self.text, usage_metadata=None)


def test_gemini_extractor_with_fake_client():
    models = FakeModels('{"doc_type": "invoice", "total": 978.12, "line_items": []}')
    ext = GeminiExtractor(model="m", api_key="x", rpm=6000, client=SimpleNamespace(models=models))
    res = ext.extract(Payload(b"img", "image/jpeg"))
    assert res.document.total == 978.12 and res.model == "gemini/m"
    _, contents, config = models.calls[0]
    assert config.temperature == 0 and config.response_mime_type == "application/json"


def test_daily_quota_fails_fast(monkeypatch):
    from google.genai import errors

    class QuotaModels:
        def generate_content(self, **kw):
            raise errors.APIError(429, {"error": {"code": 429, "message": "quota", "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "45606s"}]}})

    monkeypatch.setattr("time.sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))
    ext = GeminiExtractor(model="m", api_key="x", rpm=6000, client=SimpleNamespace(models=QuotaModels()))
    with pytest.raises(ExtractionError, match="429: quota exhausted"):
        ext.extract(Payload(b"img", "image/jpeg"))


@pytest.mark.parametrize("printed, order, expected", [
    ("03/18/2017", "DMY", "2017-03-18"),   # unambiguous: 18 can only be the day
    ("18.03.2017", "MDY", "2017-03-18"),
    ("10/01/2014", "MDY", "2014-10-01"),   # ambiguous: the source's convention decides
    ("10/01/2014", "DMY", "2014-01-10"),
    ("2014-10-01", "DMY", "2014-10-01"),
    ("05/10/21", "MDY", "2021-05-10"),
    ("18 March 2017", "MDY", "2017-03-18"),
    ("31/02/2017", "DMY", None),
    (None, "DMY", None),
])
def test_resolve_date(printed, order, expected):
    from invoice_extractor.normalize import resolve_date
    assert resolve_date(printed, order) == expected


def test_finalize_sets_date_from_printed():
    from invoice_extractor.postprocess import finalize
    doc = Document(doc_type="invoice", date_printed="10/01/2014", date="1999-01-01")
    assert finalize(doc, "MDY").date == "2014-10-01"
    assert "date" not in gemini_json_schema()["properties"]


def test_label_fixes_are_well_formed():
    from eval.datasets import LABEL_FIXES
    for sample_id, fixes in LABEL_FIXES.items():
        if sample_id.startswith("_"):
            continue
        for fix in fixes.values():
            assert {"value", "label", "evidence"} <= fix.keys()


def test_service_charge_is_part_of_the_totals_check():
    # CORD test-014: 238,000 + service 17,850 + tax 25,585 = 281,435
    doc = _doc(subtotal=238000, service_charge=17850, tax=25585, total=281435)
    assert check(doc) == []
    assert {i.code for i in check(_doc(subtotal=238000, tax=43435, total=300000))} == {"totals_math"}


def test_line_discounts_before_or_after_subtotal():
    items = [LineItem(description="a", quantity=2, unit_price=9000, total=18000, discount=-7200)]
    assert check(_doc(line_items=items, subtotal=18000, total=18000)) == []
    assert check(_doc(line_items=items, subtotal=10800, total=10800)) == []


def test_map_cord_keeps_priced_addons_only():
    gt = {"menu": {"nm": "Lemon Tea", "cnt": "1", "price": "25.000", "sub": [
        {"nm": "Extra Jelly", "cnt": "1", "price": "5.000"}, {"nm": "- Less ice", "price": "0"}]},
        "sub_total": {"service_price": "1.000", "discount_price": "2.000"}}
    truth, items = map_cord(gt)
    assert [i.description for i in items] == ["Lemon Tea", "Extra Jelly"]
    assert truth == {"service_charge": 1000, "discount": -2000}


def test_options_are_not_scored_or_summed():
    items = [LineItem(description="Black Sakura", quantity=1, total=59091),
             LineItem(description="Longan", quantity=1, total=0, is_option=True)]
    doc = _doc(line_items=items, subtotal=59091, total=59091)
    assert check(doc) == []
    s = score_document(doc, {"total": 59091}, [LineItem(description="Black Sakura", quantity=1, total=59091)])
    assert s.exact and s.items_pred == 1
