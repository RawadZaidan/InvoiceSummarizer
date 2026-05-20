"""
Smart Invoice / Purchase Order Scanner
=======================================
Hybrid pipeline:
  1. Native PDF text layer extraction (pdfplumber) — fast, lossless
  2. Fallback: PDF → image rendering (pdf2image / pymupdf) for scanned PDFs
  3. Vision LLM via OpenRouter (Qwen2.5-VL-72B) for full document understanding
     — handles Arabic, French, English, mixed layouts, stamps, handwriting

Dependencies:
    python -m pip install -r requirements.txt

Usage:
    python invoice_scanner.py path/to/file.pdf
    python invoice_scanner.py path/to/scan.jpg
    python invoice_scanner.py ./invoices/          # batch mode (folder)
"""

import os
import sys
import json
import base64
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY not found. Add it to your .env file:\n"
        "  OPENROUTER_API_KEY=sk-or-..."
    )

# Best vision model for multilingual OCR + document understanding on OpenRouter.
# Qwen2.5-VL-72B: top DocVQA / multilingual OCR scores, supports Arabic RTL text.
# Fallback options (uncomment to swap):
#   "google/gemini-2.0-flash-001"            — fast, good multilingual
#   "anthropic/claude-3.5-sonnet"             — excellent structured extraction
#   "meta-llama/llama-3.2-90b-vision-instruct"
# VISION_MODEL = "qwen/qwen2.5-vl-72b-instruct"
VISION_MODEL = "google/gemini-2.5-flash-lite"

# Native-text PDFs: we try text extraction first; only call vision if
# extracted text is suspiciously short (likely a scanned/image PDF).
MIN_TEXT_LENGTH = 80   # characters; below this → treat as scanned

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ──────────────────────────────────────────────────────────────────────────────
# Extraction prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert document intelligence system specialized in invoices and purchase orders from Lebanon and the MENA region. You handle documents in Arabic, French, English, or any combination.

Your task: analyze the provided document (image or text) and extract EXACTLY these fields as valid JSON.

Return ONLY a JSON object — no markdown, no explanation, no backticks. Example:
{
  "document_type": "invoice",
  "invoice_number": "5987",
  "invoice_date": "2019-01-04",
  "issued_by": {
    "name": "Brand Name"
  },
  "billed_to": {
    "name": "Dayanne S.Clone",
    "address": "B- Unknown Street",
    "city": "Location",
    "country": "Lorem Ipsum"
  },
  "line_items": [
    {
      "description": "Product Name",
      "qty": 4,
      "unit": null,
      "unit_price": 50.00,
      "subtotal": 200.00,
      "tax_rate": null
    }
  ],
  "totals": {
    "subtotal": 545.00,
    "tax": 0.00,
    "total": 545.00,
    "currency": "USD"
  },
  "payment_terms": "Net 30",
  "notes": "Delivery within 5 business days.",
  "terms_and_conditions": "All sales are final.",
  "confidence": 0.97
}

Field rules:
- document_type: exactly "invoice" or "purchase_order" (use context clues like "فاتورة"/"Invoice"/"Facture" vs "أمر شراء"/"PO"/"Bon de commande")
- invoice_number: the invoice or PO reference number as a string; null if not found
- invoice_date: ISO format YYYY-MM-DD when possible; null if not found
- issued_by.name: the seller/vendor/issuer name; preserve original script; null if not found
- billed_to.name: the customer/buyer name; preserve original script; null if not found
- billed_to.address: street address of the buyer; null if not found
- billed_to.city: city of the buyer; null if not found
- billed_to.country: country of the buyer; null if not found
- line_items: list of every line item on the document
  - description: product or service name
  - qty: numeric quantity; null if not present
  - unit: unit of measure (e.g. "kg", "pcs"); null if not present
  - unit_price: numeric unit price; null if not present
  - subtotal: numeric line subtotal (qty × unit_price); null if not present
  - tax_rate: tax rate as a percentage string (e.g. "11%") or null
- totals.subtotal: numeric sum before tax; null if not found
- totals.tax: numeric tax amount; null if not found
- totals.total: numeric grand total; null if not found
- totals.currency: ISO currency code of the totals (USD, LBP, EUR…); null if unclear
- payment_terms: payment terms text (e.g. "Net 30", "Due on receipt", "30 jours fin de mois", "دفع فوري"); null if not found
- notes: any free-text notes, remarks, or comments on the document not captured elsewhere; null if not found
- terms_and_conditions: terms and conditions text printed on the document; null if not found
- confidence: float 0.0–1.0 reflecting your overall extraction confidence

If a field is genuinely not present, use null. Never guess wildly.
"""

USER_PROMPT_IMAGE = "Please extract the invoice/PO data from this document image."
USER_PROMPT_TEXT = lambda text: f"""Please extract the invoice/PO data from the following document text.
The text was extracted via OCR/PDF parser — there may be minor formatting noise.

--- DOCUMENT TEXT START ---
{text}
--- DOCUMENT TEXT END ---"""


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Try native PDF text extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    """Extract embedded text from a PDF using pdfplumber (most accurate for text PDFs)."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts).strip()
    except ImportError:
        pass

    # Fallback: pymupdf (fitz) — faster, also handles Arabic with proper RTL
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        pages_text = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages_text).strip()
    except ImportError:
        pass

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Convert PDF page to image (for scanned PDFs)
# ──────────────────────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """
    Render each PDF page to a PNG image.
    Returns list of temp file paths. Caller is responsible for cleanup.
    Tries pdf2image first, then pymupdf as fallback.
    """
    tmp_paths = []

    # Try pdf2image (poppler-based) — best quality
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=dpi)
        for i, img in enumerate(images):
            tmp = tempfile.NamedTemporaryFile(suffix=f"_page{i}.png", delete=False)
            tmp.close()
            img.save(tmp.name, "PNG")
            tmp_paths.append(tmp.name)
        return tmp_paths
    except (ImportError, Exception):
        pass

    # Fallback: pymupdf
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            tmp = tempfile.NamedTemporaryFile(suffix=f"_page{i}.png", delete=False)
            tmp.close()
            pix.save(tmp.name)
            tmp_paths.append(tmp.name)
        doc.close()
        return tmp_paths
    except ImportError:
        pass

    raise RuntimeError(
        "Cannot render PDF to image. Install either pdf2image+poppler or pymupdf:\n"
        "  pip install pdf2image pymupdf"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Encode image as base64 for the vision API
# ──────────────────────────────────────────────────────────────────────────────

def image_to_base64(image_path: str) -> tuple[str, str]:
    """Returns (base64_string, mime_type)."""
    suffix = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    mime = mime_map.get(suffix, "image/png")

    # Optionally resize very large images to stay within token budgets
    try:
        from PIL import Image as PILImage
        import io
        img = PILImage.open(image_path)
        max_side = 2048
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        buf = io.BytesIO()
        fmt = "JPEG" if mime == "image/jpeg" else "PNG"
        img.save(buf, format=fmt, quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

    return b64, mime


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Call OpenRouter vision/text LLM
# ──────────────────────────────────────────────────────────────────────────────

def call_llm_with_image(image_path: str) -> dict:
    """Send an image to the vision LLM and return parsed JSON result."""
    b64, mime = image_to_base64(image_path)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": USER_PROMPT_IMAGE},
                ],
            },
        ],
        max_tokens=1024,
        temperature=0.0,   # deterministic extraction
    )
    return _parse_response(response.choices[0].message.content)


def call_llm_with_text(text: str) -> dict:
    """Send extracted text to the LLM and return parsed JSON result."""
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEXT(text)},
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    return _parse_response(response.choices[0].message.content)


def _parse_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Extract first JSON object in the response
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "document_type": None,
            "invoice_number": None,
            "invoice_date": None,
            "issued_by": {"name": None},
            "billed_to": {"name": None, "address": None, "city": None, "country": None},
            "line_items": [],
            "totals": {"subtotal": None, "tax": None, "total": None, "currency": None},
            "payment_terms": None,
            "notes": None,
            "terms_and_conditions": None,
            "confidence": 0.0,
            "_raw_response": raw,
            "_error": "Failed to parse LLM JSON response",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main processing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_file(file_path: str) -> dict:
    """
    Full hybrid pipeline:
      PDF with text  → text extraction → LLM (text mode)
      PDF scanned    → render to image → LLM (vision mode)
      Image file     → LLM (vision mode)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    result = {}
    strategy_used = ""

    if suffix == ".pdf":
        # Try native text first
        print(f"  [1/2] Extracting native PDF text from {path.name}…")
        pdf_text = extract_pdf_text(str(path))

        if len(pdf_text) >= MIN_TEXT_LENGTH:
            strategy_used = "native_text_llm"
            print(f"  [2/2] Sending text ({len(pdf_text)} chars) to LLM…")
            result = call_llm_with_text(pdf_text)
        else:
            # Scanned PDF — render to images and use vision
            strategy_used = "scanned_pdf_vision"
            print(f"  [2/2] PDF appears scanned, rendering to images…")
            tmp_images = []
            try:
                tmp_images = pdf_to_images(str(path))
                if not tmp_images:
                    raise RuntimeError("No pages rendered from PDF.")

                # Process first page (usually enough for header/total detection)
                # For multi-page invoices, we merge results from all pages
                results_per_page = []
                for i, img_path in enumerate(tmp_images):
                    print(f"       Page {i+1}/{len(tmp_images)} → vision LLM…")
                    page_result = call_llm_with_image(img_path)
                    results_per_page.append(page_result)

                result = _merge_page_results(results_per_page)
            finally:
                for tmp in tmp_images:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

    elif suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}:
        strategy_used = "image_vision"
        print(f"  [1/1] Sending image to vision LLM…")
        result = call_llm_with_image(str(path))

    else:
        raise ValueError(
            f"Unsupported file type: {suffix}\n"
            "Supported: .pdf, .jpg, .jpeg, .png, .webp, .bmp, .tiff"
        )

    result["_file"] = path.name
    result["_strategy"] = strategy_used
    return result


def process_file_timed(file_path: str) -> dict:
    t0 = time.perf_counter()
    result = process_file(file_path)
    result["_elapsed_s"] = round(time.perf_counter() - t0, 2)
    return result


def _merge_page_results(results: list[dict]) -> dict:
    """
    Merge multi-page extraction results.
    Priority: highest-confidence page wins for scalar fields.
    Line items: concatenated from all pages.
    Totals: last page with a non-null total wins (grand totals appear on the last page).
    """
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    best = max(results, key=lambda r: r.get("confidence", 0))
    merged = dict(best)

    # Collect all line items across pages
    all_items = []
    for r in results:
        all_items.extend(r.get("line_items") or [])
    merged["line_items"] = all_items

    # Use totals from the last page that has a non-null total value
    for r in reversed(results):
        t = r.get("totals") or {}
        if t.get("total") is not None:
            merged["totals"] = t
            break

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Pretty output
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(val, prefix="", suffix="") -> str:
    if val is None:
        return "—"
    return f"{prefix}{val}{suffix}"


def print_result(result: dict):
    """Print a clean, human-readable summary."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {result.get('_file', 'Unknown file')}")
    print(sep)

    # Invoice Details
    print(f"  Invoice #     : {_fmt(result.get('invoice_number'))}")
    print(f"  Invoice Date  : {_fmt(result.get('invoice_date'))}")
    print(f"  Type          : {_fmt(result.get('document_type'))}")

    # Issued By
    issued = result.get('issued_by') or {}
    print(f"\n  Issued By")
    print(f"    Name        : {_fmt(issued.get('name'))}")

    # Billed To
    billed = result.get('billed_to') or {}
    print(f"\n  Billed To")
    print(f"    Name        : {_fmt(billed.get('name'))}")
    print(f"    Address     : {_fmt(billed.get('address'))}")
    print(f"    City        : {_fmt(billed.get('city'))}")
    print(f"    Country     : {_fmt(billed.get('country'))}")

    # Line Items
    items = result.get('line_items') or []
    print(f"\n  Line Items ({len(items)})")
    if items:
        print(f"  {'#':>3}  {'Description':<24}  {'Qty':>5}  {'Unit':<6}  {'Unit Price':>10}  {'Subtotal':>10}  {'Tax Rate':>8}")
        print(f"  {'─'*3}  {'─'*24}  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}")
        for i, item in enumerate(items, 1):
            desc = str(item.get('description') or '—')[:24]
            qty  = _fmt(item.get('qty'))
            unit = _fmt(item.get('unit'))[:6]
            up   = _fmt(item.get('unit_price'))
            sub  = _fmt(item.get('subtotal'))
            tax  = _fmt(item.get('tax_rate'))
            print(f"  {i:>3}  {desc:<24}  {qty:>5}  {unit:<6}  {up:>10}  {sub:>10}  {tax:>8}")
    else:
        print(f"    (none found)")

    # Totals
    totals = result.get('totals') or {}
    curr = totals.get('currency') or ''
    print(f"\n  Totals")
    print(f"    Subtotal    : {_fmt(totals.get('subtotal'))} {curr}".rstrip())
    print(f"    Tax         : {_fmt(totals.get('tax'))} {curr}".rstrip())
    print(f"    Total       : {_fmt(totals.get('total'))} {curr}".rstrip())

    # Payment terms / notes / T&C
    payment_terms = result.get('payment_terms')
    notes = result.get('notes')
    tandc = result.get('terms_and_conditions')
    if payment_terms or notes or tandc:
        print(f"\n  Payment & Terms")
        if payment_terms:
            print(f"    Payment Terms : {payment_terms}")
        if notes:
            print(f"    Notes         : {notes}")
        if tandc:
            print(f"    T&C           : {tandc}")

    conf = result.get('confidence')
    conf_str = f"{conf:.0%}" if isinstance(conf, (float, int)) else "—"
    elapsed = result.get('_elapsed_s')
    elapsed_str = f"{elapsed}s" if elapsed is not None else "—"
    print(f"\n  Confidence    : {conf_str}")
    print(f"  Strategy      : {result.get('_strategy') or '—'}")
    print(f"  Time          : {elapsed_str}")
    if result.get("_error"):
        print(f"  ⚠ Error       : {result['_error']}")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    all_results = []

    if target.is_dir():
        # Batch mode
        supported = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
        files = [f for f in sorted(target.iterdir()) if f.suffix.lower() in supported]
        if not files:
            print(f"No supported files found in {target}")
            sys.exit(1)
        print(f"\nBatch mode: processing {len(files)} file(s) in '{target}'\n")
        for f in files:
            print(f"\n→ {f.name}")
            try:
                result = process_file_timed(str(f))
                print_result(result)
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ Error: {e}")
                all_results.append({"_file": f.name, "_error": str(e)})
    else:
        # Single file mode
        result = process_file_timed(str(target))
        print_result(result)
        all_results.append(result)

    # Save all results to JSON
    out_path = "invoice_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to {out_path}\n")

    return all_results


if __name__ == "__main__":
    main()
