# 🧾 Invoice & Receipt Extractor

Turn scanned invoices and receipts into validated, structured data with a **free LLM API**
(Gemini Flash), and measure how far you can trust it on **public labelled datasets**.

**TR —** Taranmış fatura ve fişleri ücretsiz bir LLM API'si (Gemini Flash) ile sabit bir şemaya
göre yapılandırılmış veriye çeviren bir proje. Modelin ne kadar güvenilir olduğu, etiketli açık
veri setlerinde ölçülüyor. Okuma işini LLM yapıyor, okunan değerlerin tutarlı olup olmadığını kod
kontrol ediyor. Tüm kontrollerden geçen belge otomatik onaylanıyor, geçemeyen insan kontrolüne
gidiyor.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Gemini](https://img.shields.io/badge/LLM-Gemini%20Flash%20(free%20tier)-4285F4)
![Pydantic](https://img.shields.io/badge/Schema-Pydantic-E92063)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

```
 image / PDF ──► loader ──► LLM extractor ──► Document (Pydantic) ──► validation ──► auto-accept
                (resize,     Gemini Flash,      fixed schema,          line math,       or
                 PDF pages)  JSON-schema output  typed amounts          totals, IBAN,     human review
                             (or local Ollama)                          dates
```

| Component | What it does |
|---|---|
| `invoice_extractor/schema.py` | One Pydantic schema (vendor, ids, date, IBAN, line items, subtotal/tax/total) used for generation, validation and scoring |
| `invoice_extractor/extractors.py` | Gemini with **schema-constrained JSON output**, free-tier rate limiting and 429 retry; optional local Ollama vision model for comparison |
| `invoice_extractor/validation.py` | Deterministic checks: qty × price = line total, lines = subtotal, subtotal + tax = total, IBAN mod-97, date sanity |
| `eval/` | Maps two public datasets onto the schema, caches every prediction, scores fields and line items, measures the review routing |
| `app/streamlit_app.py` | Upload → review and correct → export CSV/JSON, with a per-document auto-accept / review status |

## Evaluation

| dataset | documents | labelled fields |
|---|---|---|
| [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC BY 4.0) | 100 Indonesian receipts (test split) | line items, subtotal, tax, total |
| [katanaml invoices](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1) | 26 synthetic invoices (test split) | number, date, seller/client + tax ids, IBAN, line items, totals |

Scoring rules (`eval/metrics.py`):
- **Amounts** must match to the cent; **ids** match on letters and digits; **dates** as ISO strings;
  **names** may be a prefix of a label that also contains the address ("Clay-Koch" ✓, "Clay" ✗).
- **Line items** are paired one-to-one by description similarity; a line is correct only if every
  labelled number (quantity, unit price, total) matches. Precision, recall and F1 over lines.
- **Fully correct document**: every labelled field and every line right, and no extra lines.
- **Review routing**: of the documents with no validation issue, how many are fully correct
  (auto-accept precision), and how many wrong documents were flagged.

### Results

**Model: `gemini-3.1-flash-lite`** (free tier, temperature 0, schema-constrained JSON).
`gemini-3.5-flash` is just as accurate but its free tier is only **20 requests/day**, too few to
evaluate on; the lite model is the one you can actually run the whole eval on for free.

| dataset | field accuracy (micro) | line-item F1 | fully correct | auto-accepted · of which correct · wrong docs flagged | median latency |
|---|:--:|:--:|:--:|:--:|:--:|
| invoices (26) | **100%** | 100% | **100%** | 100% · 100% · – | 4.4 s |
| CORD receipts (100) | **99.1%** | 97.1% | **91%** | 96% · 94% · 33% | 3.1 s |

`–` on invoices: with the label fixes below, no invoice is wrong, so there is nothing to flag.

**The results are reproduced from cached predictions in `eval/predictions/`**, so you can rescore
them without an API key: `python -m eval.run_eval --dataset cord` reads the cache and prints the
table. Add `--rerun` to call the API again.

#### What moved the numbers (iteration, not a single shot)
Reading every wrong field — not just the score — drove four fixes, each with a regression test:

1. **Dates are resolved by code, not the LLM.** `10/01/2014` is genuinely ambiguous; asking the
   model to emit ISO dates made it guess day/month order (8 invoices wrong). Now the model copies
   the printed date verbatim and `resolve_date()` applies the *source's* convention (MDY for these
   US invoices). Invoice dates went 73% → 100%.
2. **Options vs. purchases.** Receipt modifiers ("− Less ice", a bundle's flavours) were extracted
   as zero-price line items, wrecking item precision. An `is_option` flag separates choices from
   things bought; options are excluded from scoring and from the sum checks. Item F1 93% → 97%.
3. **Service charge and discounts are first-class.** `subtotal + tax = total` is wrong for receipts
   with a service charge or a discount. The schema and the checks now model both, and accept either
   a tax-exclusive or tax-inclusive printing. Fewer false "totals_math" flags.
4. **Line discounts.** A line's `total` is the pre-discount amount; a printed line discount goes in
   `discount`. The subtotal check accepts the subtotal before *or* after line discounts.

<details><summary>Before → after on CORD (100 receipts)</summary>

| | fully correct | item F1 |
|---|:--:|:--:|
| v1 (first prompt) | 77% | 92.7% |
| v2 (+ service/discount schema, prompt) | 85% | 93.9% |
| v3 (+ is_option) | **91%** | **97.1%** |
</details>

### Known limits of the checks
- **Ground-truth errors.** The checks also fire on some labels. I verified these against the
  images and recorded them in [`eval/label_fixes.json`](eval/label_fixes.json): **7 of 26 invoices**
  have a wrong IBAN (OCR `S↔5`, `I↔1`, `O↔0`, repeated digit groups), one has the seller tax id in
  the name field, and one has a wrong total. The table above uses the corrected labels; add
  `--raw-labels` to score against the originals (invoices then read as 96.5% / 69% fully correct —
  i.e. the extractor is *right* on most of what the labels get wrong).
- **5 of 100 CORD receipts have no printed total** at all, so they are flagged for review by design.
- **Misattribution the checks can't see.** When a receipt is internally consistent, a value copied
  from the wrong spot still adds up. That is why ~6% of auto-accepted documents are still imperfect;
  catching those needs a second reader, not arithmetic.

## Run it

```bash
py -3.14 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env            # then paste your free key from https://aistudio.google.com/apikey

python scripts/list_models.py          # which Flash models your key can use
python -m eval.run_eval --dataset cord # rescore cached predictions (no key needed)
python -m eval.run_eval --dataset cord --rerun   # call the API again
streamlit run app/streamlit_app.py
pytest -q
```

Predictions are cached in `eval/predictions/<model>/`, so a run that hits the free-tier daily
limit can be restarted and continues where it stopped. Note that free-tier inputs may be used
by Google to improve its products, so don't send real customer documents.

Local comparison (no API): `ollama pull qwen2.5vl:3b`, then
`python -m eval.run_eval --provider ollama`.

## License
MIT. Datasets keep their own licenses (see the dataset cards).
