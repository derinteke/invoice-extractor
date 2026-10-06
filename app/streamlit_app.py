"""Streamlit UI: upload invoices/receipts, review the extraction, export the results.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invoice_extractor.extractors import ExtractionError, get_extractor  # noqa: E402
from invoice_extractor.loaders import load_payload, pdf_to_images  # noqa: E402
from invoice_extractor.postprocess import finalize  # noqa: E402
from invoice_extractor.schema import Document, LineItem  # noqa: E402
from invoice_extractor.validation import check  # noqa: E402

load_dotenv()
st.set_page_config(page_title="Invoice Extractor", page_icon="🧾", layout="wide")
state = st.session_state
state.setdefault("results", {})   # file hash -> {"name", "document", "model", "latency"}


@st.cache_resource
def extractor_for(provider: str, model: str):
    return get_extractor(provider, model or None)


@st.cache_data
def sample_files() -> list[tuple[str, bytes]]:
    from eval.datasets import load_samples

    picks = [("invoices", 0), ("invoices", 1), ("cord", 3), ("cord", 7)]
    out = []
    for name, idx in picks:
        sample = list(load_samples(name, limit=idx + 1))[idx]
        out.append((f"{sample.id}.jpg", sample.payload.data))
    return out


def preview(data: bytes, name: str) -> Image.Image:
    if name.lower().endswith(".pdf"):
        return pdf_to_images(data, max_pages=1)[0]
    return Image.open(io.BytesIO(data))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", ["gemini", "ollama"],
                            help="gemini = free API tier; ollama = local vision model")
    model = st.text_input("Model name (empty = default from .env)")
    st.caption("Gemini reads `GEMINI_API_KEY` from `.env`.")
    date_order = st.radio("Ambiguous dates like 10/01/2024 are", ["DMY", "MDY"], horizontal=True,
                          format_func={"DMY": "day/month (TR, EU)", "MDY": "month/day (US)"}.get)
    st.divider()
    use_samples = st.toggle("Use sample documents", help="4 labelled documents from the eval sets")

st.title("🧾 Invoice & Receipt Extractor")
st.write("Upload scanned invoices or receipts. An LLM reads them into a fixed schema, and "
         "deterministic checks decide whether each one can be auto-accepted or needs review.")

files: list[tuple[str, bytes]] = []
if use_samples:
    files = sample_files()
else:
    uploads = st.file_uploader("Documents", type=["png", "jpg", "jpeg", "webp", "pdf"],
                               accept_multiple_files=True)
    files = [(u.name, u.getvalue()) for u in uploads or []]

# ---------------------------------------------------------------- extraction
pending = [(n, d) for n, d in files if hashlib.sha1(d).hexdigest() not in state.results]
if files and st.button(f"Extract {len(pending)} new document(s)", type="primary",
                       disabled=not pending):
    try:
        extractor = extractor_for(provider, model)
    except ExtractionError as exc:
        st.error(str(exc))
        st.stop()
    progress = st.progress(0.0)
    for i, (name, data) in enumerate(pending, 1):
        try:
            res = extractor.extract(load_payload(data, name))
            state.results[hashlib.sha1(data).hexdigest()] = {
                "name": name, "document": finalize(res.document, date_order).model_dump(),
                "model": res.model, "latency": res.latency_s,
            }
        except ExtractionError as exc:
            st.error(f"{name}: {exc}")
        progress.progress(i / len(pending))

for name, data in files:
    key = hashlib.sha1(data).hexdigest()
    if key not in state.results:
        continue
    entry = state.results[key]
    doc = Document.model_validate(entry["document"])
    issues = check(doc)
    status = "✅ auto-accept" if not issues else f"⚠️ review ({len(issues)} issue(s))"

    with st.expander(f"{name} — {status}", expanded=True):
        left, right = st.columns([2, 3])
        left.image(preview(data, name), width="stretch")
        with right:
            st.caption(f"{entry['model']} · {entry['latency']:.1f} s")
            header = {k: v for k, v in doc.model_dump().items() if k != "line_items"}
            edited_header = st.data_editor(
                pd.DataFrame([header]).T.rename(columns={0: "value"}).astype(str),
                key=f"h-{key}", width="stretch",
            )
            items = pd.DataFrame([it.model_dump() for it in doc.line_items],
                                 columns=list(LineItem.model_fields))
            edited_items = st.data_editor(items, key=f"i-{key}", num_rows="dynamic",
                                          width="stretch")
            for issue in issues:
                st.warning(issue.message)

            if st.button("Save corrections", key=f"s-{key}"):
                values = {k: (None if v in ("None", "") else v)
                          for k, v in edited_header["value"].items()}
                rows = edited_items.where(pd.notna(edited_items), None).to_dict("records")
                try:
                    entry["document"] = Document.model_validate(
                        {**values, "line_items": rows}).model_dump()
                    st.rerun()
                except ValueError as exc:
                    st.error(f"Invalid value: {exc}")

# ---------------------------------------------------------------- history & export
if state.results:
    st.divider()
    st.subheader("Processed documents")
    table = pd.DataFrame([
        {"file": e["name"], **{k: e["document"][k] for k in
                               ("doc_type", "vendor_name", "date", "currency", "total")},
         "items": len(e["document"]["line_items"]),
         "status": "review" if check(Document.model_validate(e["document"])) else "auto-accept"}
        for e in state.results.values()
    ])
    st.dataframe(table, width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(table))
    c2.metric("Auto-accepted", f"{(table['status'] == 'auto-accept').mean():.0%}")
    totals = table.dropna(subset=["total"])
    if not totals.empty:
        by_cur = totals.groupby(totals["currency"].fillna("?"))["total"].sum()
        c3.metric("Total by currency", " · ".join(f"{v:,.2f} {k}" for k, v in by_cur.items()))
        chart = totals.assign(vendor=totals["vendor_name"].fillna(totals["file"]))
        st.bar_chart(chart, x="vendor", y="total", color="currency")

    d1, d2 = st.columns(2)
    d1.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"),
                       "documents.csv", "text/csv")
    d2.download_button(
        "Download JSON",
        json.dumps([{"file": e["name"], **e["document"]} for e in state.results.values()],
                   ensure_ascii=False, indent=2).encode("utf-8"),
        "documents.json", "application/json",
    )
