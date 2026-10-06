"""Output schema shared by every extractor, the validator and the evaluation."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(description="Item name exactly as printed, without the price.")
    is_option: bool = Field(
        default=False,
        description="True for a choice printed under another item without its own charge "
                    "(size, topping, spice level, 'less ice'). False for anything purchased.",
    )
    quantity: float | None = Field(default=None, description="Number of units, e.g. 2 or 1.5.")
    unit_price: float | None = Field(default=None, description="Price of one unit, before tax.")
    total: float | None = Field(
        default=None,
        description="Amount of this line before tax and before any line discount "
                    "(normally quantity x unit price).",
    )
    discount: float | None = Field(
        default=None, description="Discount printed for this line, as a negative number."
    )


def purchased(items: list[LineItem]) -> list[LineItem]:
    """Line items that are purchases, i.e. without options/modifiers."""
    return [it for it in items if not it.is_option]


class Document(BaseModel):
    doc_type: Literal["invoice", "receipt", "other"] = Field(
        description="invoice = formal bill with an invoice number; receipt = till/shop slip."
    )
    vendor_name: str | None = Field(default=None, description="Seller/company name only, no address.")
    vendor_tax_id: str | None = Field(default=None, description="Seller tax / VAT number as printed.")
    customer_name: str | None = Field(default=None, description="Buyer/client name only, no address.")
    customer_tax_id: str | None = Field(default=None, description="Buyer tax / VAT number as printed.")
    invoice_number: str | None = Field(default=None, description="Invoice or receipt number as printed.")
    date_printed: str | None = Field(
        default=None, description="Issue date copied exactly as printed, e.g. '03/18/2017'."
    )
    date: str | None = Field(
        default=None, description="ISO date, resolved by code from date_printed (not by the LLM)."
    )
    currency: str | None = Field(
        default=None, description="ISO 4217 code (USD, EUR, TRY, IDR...) if it can be inferred."
    )
    iban: str | None = Field(default=None, description="Seller IBAN without spaces.")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = Field(default=None, description="Total before tax.")
    discount: float | None = Field(
        default=None, description="Document-level discount as a negative number."
    )
    service_charge: float | None = Field(
        default=None, description="Service charge, printed separately from tax."
    )
    tax: float | None = Field(default=None, description="Total tax / VAT amount, without service charge.")
    total: float | None = Field(default=None, description="Final amount to pay, including tax.")


# Fields compared one-to-one in the evaluation (line items are scored separately).
SCALAR_FIELDS = [
    "vendor_name", "vendor_tax_id", "customer_name", "customer_tax_id",
    "invoice_number", "date", "iban", "subtotal", "discount", "service_charge", "tax", "total",
]
NUMERIC_FIELDS = {"subtotal", "discount", "service_charge", "tax", "total"}
# Filled in by code after extraction, so they are not part of the LLM's output schema.
CODE_ONLY_FIELDS = {"date"}


def gemini_json_schema() -> dict:
    """Pydantic's JSON schema with every $ref inlined (structured output is safest without refs)."""
    schema = Document.model_json_schema()
    defs = schema.pop("$defs", {})
    for name in CODE_ONLY_FIELDS:
        schema["properties"].pop(name, None)

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs[node["$ref"].split("/")[-1]])
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return inline(schema)
