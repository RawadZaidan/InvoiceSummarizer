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
VISION_MODEL = "qwen/qwen2.5-vl-72b-instruct"

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
  "date": "2024-03-15",
  "client_name": "شركة المستقبل للتجارة",
  "totals": [
    {"amount": "1,250,000 LBP", "currency": "LBP"},
    {"amount": "14.03 USD", "currency": "USD"}
  ],
  "confidence": 0.97
}

Field rules:
- document_type: exactly "invoice" or "purchase_order" (use context clues like "فاتورة"/"Invoice"/"Facture" vs "أمر شراء"/"PO"/"Bon de commande")
- date: ISO format DD-MM-YYYY when possible, or best available format; null if not found
- client_name: the customer/buyer name (not the seller/vendor); preserve original script (Arabic if Arabic, etc.)
- totals: list of ALL final grand totals found, with their respective currencies. If an invoice shows a total in LBP and its equivalent in USD, or even two amounts, include both.
  - amount: the string amount as written (e.g. "1,250,000 LBP" or "14.03")
  - currency: ISO code if identifiable (USD, LBP, EUR, SAR…); null if unclear
- confidence: float 0.0–1.0 reflecting your extraction confidence

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
            "date": None,
            "client_name": None,
            "totals": [],
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


def _merge_page_results(results: list[dict]) -> dict:
    """
    Merge multi-page extraction results.
    Priority: highest-confidence result wins per field.
    Totals: Aggregate all unique currency totals found across pages (favoring the last page).
    """
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    # Base from highest-confidence page
    best = max(results, key=lambda r: r.get("confidence", 0))
    merged = dict(best)

    # Gather totals from all pages, favoring pages near the end if they duplicate a currency
    all_totals_by_curr = {}
    
    # Process from first down to last so the last page overwrites earlier pages for the same currency
    for r in results:
        for t in r.get("totals", []):
            curr = t.get("currency")
            if curr:
                all_totals_by_curr[curr] = t
    
    # Also add whatever the best page had if it didn't get caught
    for t in merged.get("totals", []):
        curr = t.get("currency")
        if curr and curr not in all_totals_by_curr:
            all_totals_by_curr[curr] = t

    merged["totals"] = list(all_totals_by_curr.values())

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Pretty output
# ──────────────────────────────────────────────────────────────────────────────

def print_result(result: dict):
    """Print a clean, human-readable summary."""
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  📄  {result.get('_file', 'Unknown file')}")
    print(sep)
    print(f"  Type          : {result.get('document_type') or '—'}")
    print(f"  Date          : {result.get('date') or '—'}")
    print(f"  Client Name   : {result.get('client_name') or '—'}")
    
    totals = result.get('totals') or []
    if not totals:
        print(f"  Totals        : —")
    else:
        for i, t in enumerate(totals):
            prefix = "  Totals        :" if i == 0 else "                 "
            print(f"{prefix} {t.get('amount')} ({t.get('currency', '—')})")

    conf = result.get('confidence')
    conf_str = f"{conf:.0%}" if isinstance(conf, (float, int)) else "—"
    print(f"  Confidence    : {conf_str}")
    print(f"  Strategy      : {result.get('_strategy') or '—'}")
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
                result = process_file(str(f))
                print_result(result)
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ Error: {e}")
                all_results.append({"_file": f.name, "_error": str(e)})
    else:
        # Single file mode
        result = process_file(str(target))
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
