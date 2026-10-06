# Invoice & Receipt Extractor

Bu projede taranmış fatura ve fişleri ücretsiz bir LLM API'siyle (Gemini) sabit bir şemaya
çeviriyorum ve sonucun ne kadar güvenilir olduğunu etiketli açık veri setlerinde ölçüyorum.
Belgeyi model okuyor, okunan değerlerin birbirini tutup tutmadığını ise kod kontrol ediyor;
tutmayan belge insan kontrolüne gidiyor. 26 faturada tüm alanlar doğru çıktı, 100 gerçek fişin
91'i ise eksiksiz okundu. Faturalardaki hataların çoğunun aslında veri setinin etiketlerinden
geldiğini görsellerle doğrulayıp ayrıca kaydettim.

---

I built this to see how far a free LLM tier can go on a boring but very real task: reading
scanned invoices and receipts into structured data. The model reads the document into a fixed
Pydantic schema, and a set of plain arithmetic checks decides whether the result can be accepted
automatically or should go to a person.

The part I cared about most is the evaluation. Everything is scored against two public labelled
datasets, every prediction is cached in the repo, and the numbers below can be reproduced
without an API key.

```
image / PDF -> loader -> Gemini (JSON schema) -> Document -> code post-processing -> checks -> accept / review
```

## How it works

The schema in `invoice_extractor/schema.py` covers what you would expect on an invoice: seller
and buyer names and tax ids, invoice number, date, IBAN, line items, subtotal, discount, service
charge, tax and total. The same schema is used to constrain the model's output, to validate it,
and to score it.

`invoice_extractor/extractors.py` sends the image to Gemini with a JSON schema, so the reply is
always parseable. It also respects the free tier's per-minute limit and stops cleanly when the
daily quota runs out instead of waiting for hours. There is an optional local path through an
Ollama vision model if you want to compare against something that runs offline.

`invoice_extractor/validation.py` holds the checks: quantity times unit price should equal the
line total, the lines should add up to the subtotal, subtotal plus service charge, discount and
tax should give the total, the IBAN should pass its mod-97 checksum, and the date should make
sense. Receipts print these in different ways (tax included or added on top, subtotal before or
after line discounts), so the checks accept each of those conventions.

The Streamlit app in `app/` lets you upload documents, fix any field by hand and export the
results as CSV or JSON.

## Evaluation

I used two datasets from Hugging Face:

- [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC BY 4.0), 100 Indonesian
  shop receipts from the test split, labelled with line items, subtotal, discount, service charge,
  tax and total.
- [katanaml invoices](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1), 26
  synthetic English invoices from the test split, labelled with invoice number, date, seller and
  client with their tax ids, IBAN, line items and totals.

Only the fields a dataset actually labels are scored. Amounts have to match to the cent, ids are
compared on letters and digits, dates as ISO strings. A seller name counts as correct if the label
starts with it, because the labels include the address ("Clay-Koch" is right, "Clay" is not).
Line items are paired by description and a pair only counts if every labelled number matches. A
document is fully correct when every labelled field and every line is right and there are no
extra lines.

### Results

Model: `gemini-3.1-flash-lite` on the free tier, temperature 0.

| dataset | field accuracy | line-item F1 | fully correct documents | median latency |
|---|:--:|:--:|:--:|:--:|
| invoices (26) | 100% | 100% | 100% | 4.4 s |
| CORD receipts (100) | 99.1% | 97.1% | 91% | 3.1 s |

On CORD, 96% of the receipts passed every check and were auto-accepted, and 93.8% of those were
fully correct. So roughly 6% of what the checks let through still has a mistake in it. These are
mostly cases where a value is internally consistent but taken from the wrong place, which
arithmetic alone cannot catch.

I also tried `gemini-3.5-flash`. It read the first invoice perfectly, but its free tier allows
only about 20 requests a day, which is not enough to run this evaluation, so I did not score it.

To rescore the cached predictions (no key needed):

```bash
python -m eval.run_eval --dataset cord
python -m eval.run_eval --dataset invoices
```

### What changed along the way

The first CORD run had 73% of receipts fully correct. I read every wrong field rather than just
looking at the score, and most errors fell into a few groups that each had a clear fix.

Dates were the first one. A date like `10/01/2014` is ambiguous, and when I asked the model for an
ISO date it guessed the day and month order, getting 7 of the 26 invoice dates wrong. Now the model
copies the date exactly as printed and code resolves it using the convention of the document's
source (month/day for these US-style invoices, day/month by default). Invoice dates went from
73.1% to 100%.

On receipts, the model was adding the service charge into the tax because the schema had no place
for it, and it treated price lines after a line discount inconsistently. Adding service charge and
discount fields, and defining a line total as the amount before its discount, fixed the tax field
completely.

The last group was options like "less ice" or the flavours in a bundle, which the model listed as
separate zero-priced items. I could not simply drop zero-priced lines, because some receipts have
real items that cost nothing, like a plastic bag. Instead the model now marks these with
`is_option`, and options are left out of the sums and the scoring.

All three runs below are scored with the current evaluation code, so they are directly comparable:

| run | fully correct | line-item F1 | field accuracy |
|---|:--:|:--:|:--:|
| first prompt | 73% | 93.3% | 90.2% |
| service charge and discount fields | 86% | 93.9% | 99.5% |
| `is_option` for modifiers | 91% | 97.1% | 99.1% |

Field accuracy dropped slightly in the last run because two subtotals came out wrong, while line
items improved a lot more.

### Label errors

While checking the invoice mistakes I noticed that the model was often right and the label was
wrong. I compared each case with the image and wrote them down in
[`eval/label_fixes.json`](eval/label_fixes.json) together with the evidence. Seven invoices have a
wrong IBAN in the label, with the usual OCR confusions (S and 5, I and 1, O and 0) or a repeated
group of digits, and most of those fail the IBAN checksum. One label has the seller's tax id in
the name field and one has an extra digit in the total.

The results above use the corrected labels. Scored against the original labels, the invoices come
out at 96.5% field accuracy and 69.2% fully correct, and the whole gap is these nine fields. You
can see that version with `--raw-labels`.

The checks also say something about the labels themselves. Run on the ground truth, they flag
none of the 26 corrected invoices and 7 of the 100 receipts, 5 of them because the label has no
total at all.

## Running it

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put a free key from https://aistudio.google.com/apikey into `.env`. Then:

```bash
python scripts/list_models.py
python -m eval.run_eval --dataset cord --rerun
streamlit run app/streamlit_app.py
pytest -q
```

`list_models.py` shows which Flash models your key can call, since the free model names change
over time. The tests (41) use a fake client, so they need neither a key nor a network.

Keep in mind that Google may use free-tier inputs to improve its products, so this setup is not
meant for real customer documents.

For an offline comparison you can pull a local vision model with `ollama pull qwen2.5vl:3b` and
run the evaluation with `--provider ollama`.

## License

MIT, see [LICENSE](LICENSE). The datasets keep their own licenses.
