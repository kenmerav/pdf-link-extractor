# app.py — Canva PDF → clean rows
# Extract ALL hyperlinks on chosen pages and capture the exact link-title text
# for each link (no merging), then parse Position / Type / QTY / Finish / Size.

import os, re, json, time, requests
from typing import Optional, Tuple, Dict
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ========================= App setup =========================
st.set_page_config(page_title="Canva PDF → Clean Rows", layout="wide")
st.title("📄 Canva PDF → Clean Rows (Exact Titles + Parsed Fields)")

# ========================= Parsers & helpers =========================
# Field regexes
QTY_RE     = re.compile(r"^(?:QTY|Qty|qty|Quantity)\s*:\s*([0-9Xx]+)\s*$")
FINISH_RE  = re.compile(r"^(?:FINISH|Finish)\s*:\s*(.+)$")
SIZE_RE    = re.compile(r"^(?:SIZE|Size|Dimensions?)\s*:\s*(.+)$")
TYPE_RE    = re.compile(r"^(?:TYPE|Type)\s*:\s*(.+)$")

def parse_link_title_fields(link_text: str) -> dict:
    """
    Parse a title like 'PENDANT | QTY: 1 | FINISH: Brass, White | SIZE: 32" W'
    into fields. Works with '|' and/or ';' separators. If Type isn't labeled,
    we infer it from the first unlabeled token.
    """
    fields = {"Type": "", "QTY": "", "Finish": "", "Size": ""}
    s = (link_text or "").replace("\n", " | ").strip()
    if not s:
        return fields

    # split on pipes and semicolons, normalize spaces
    parts = []
    for chunk in s.split("|"):
        subs = [x.strip() for x in chunk.split(";")]
        parts.extend([x for x in subs if x])

    for tok in parts:
        m = TYPE_RE.match(tok)
        if m and not fields["Type"]:
            fields["Type"] = m.group(1).strip(); continue
        m = QTY_RE.match(tok)
        if m and not fields["QTY"]:
            q = m.group(1).strip()
            fields["QTY"] = "" if q.upper() == "XX" else q
            continue
        m = FINISH_RE.match(tok)
        if m and not fields["Finish"]:
            fields["Finish"] = m.group(1).strip(); continue
        m = SIZE_RE.match(tok)
        if m and not fields["Size"]:
            fields["Size"] = m.group(1).strip(); continue

    # If Type not labeled, infer from the first unlabeled token
    if not fields["Type"]:
        for tok in parts:
            if ":" not in tok and tok.strip():
                fields["Type"] = tok.strip()
                break

    # Nicety: 12x24 -> 12 x 24
    if fields["Size"]:
        fields["Size"] = re.sub(r"\s*[xX]\s*", " x ", fields["Size"]).strip()

    return fields

# Position helpers
POS_AT_START = re.compile(r'^\s*(\d{1,3})[.\u2024\u00B7]?\s*')
NEXT_BULLET  = re.compile(r'\s(\d{1,3})[.\u2024\u00B7](?=\s|[A-Z])')

def split_position_and_title_start(raw: str) -> Tuple[str, str]:
    """Pull Position at the start; trim at the next bullet if we accidentally captured it."""
    s = (raw or "").strip()
    if not s:
        return "", ""
    m = POS_AT_START.match(s)
    if not m:
        return "", s
    pos = m.group(1)
    rest = s[m.end():].strip()
    m2 = NEXT_BULLET.search(rest)
    if m2:
        rest = rest[:m2.start()].strip()
    return pos, rest

def extract_link_title_strict(page, rect, pad_px: float = 4.0, band_px: float = 18.0) -> str:
    """
    Capture ONLY the words whose centers lie inside the link rectangle (± pad_px).
    If empty, use a thin horizontal band overlapping the rect (± band_px) to catch
    labels that sit just outside the box. Reading order preserved (block→line→span).
    """
    import fitz

    def norm(s: str) -> str:
        s = re.sub(r"\s*\|\s*", " | ", s)
        s = re.sub(r"\s*;\s*", "; ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    if not rect:
        return ""

    r = fitz.Rect(rect).normalize()
    R = fitz.Rect(r.x0 - pad_px, r.y0 - pad_px, r.x1 + pad_px, r.y1 + pad_px)

    raw = page.get_text("rawdict") or {}
    words = []
    # Preserve reading order: block -> line -> span
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                tx = span.get("text", "")
                bbox = span.get("bbox")
                if not tx or not bbox:
                    continue
                x0, y0, x1, y1 = bbox
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                words.append((cx, cy, x0, y0, x1, y1, tx))

    kept = [(y0, x0, tx) for cx, cy, x0, y0, x1, y1, tx in words
            if R.contains(fitz.Point(cx, cy)) and tx]

    if not kept:
        # Nearby band fallback: same y-range ± band_px, overlapping x-range
        band = fitz.Rect(r.x0, r.y0 - band_px, r.x1, r.y1 + band_px)
        kept = []
        for cx, cy, x0, y0, x1, y1, tx in words:
            if not tx:
                continue
            cy_in = (band.y0 <= cy <= band.y1)
            x_overlaps = not (x1 < r.x0 or x0 > r.x1)
            if cy_in and x_overlaps:
                kept.append((y0, x0, tx))

    if not kept:
        return ""

    kept.sort(key=lambda t: (round(t[0], 3), t[1]))
    return norm(" ".join(t[2] for t in kept))

def extract_links_by_pages(
    pdf_bytes: bytes,
    page_to_tag: Dict[int, str] | None,
    only_listed_pages: bool = True,
    pad_px: float = 4.0,
    band_px: float = 18.0,
) -> pd.DataFrame:
    """
    Read ALL hyperlinks on selected pages.
    For each link, capture ONLY the text inside that link's rectangle (strict),
    then split Position and parse Type/QTY/Finish/Size from that title.
    """
    doc = fitz.open("pdf", pdf_bytes)
    rows = []
    listed = set(page_to_tag.keys()) if page_to_tag else set()

    for pidx, page in enumerate(doc, start=1):
        if only_listed_pages and page_to_tag and pidx not in listed:
            continue
        tag_value = (page_to_tag or {}).get(pidx, "")

        for lnk in page.get_links():
            uri = (lnk.get("uri") or "").strip()
            if not uri:
                continue
            rect = lnk.get("from")

            # STRICT per-link capture
            raw_title = extract_link_title_strict(page, rect, pad_px=pad_px, band_px=band_px)
            position, title = split_position_and_title_start(raw_title)
            fields = parse_link_title_fields(title)

            rows.append({
                "page": pidx,
                "Tags": tag_value,
                "Position": position,
                "Type": fields.get("Type",""),
                "QTY": fields.get("QTY",""),
                "Finish": fields.get("Finish",""),
                "Size": fields.get("Size",""),
                "link_url": uri,     # full URL
                "link_text": title,  # exact text for this link box
            })

    return pd.DataFrame(rows)

# ========================= UI =========================
st.caption("Upload your Canva-exported PDF, map pages to Tags (room names), and extract ALL links with exact titles.")
pdf_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_extractor")

num_pages = None
if pdf_file:
    try:
        _peek = fitz.open("pdf", pdf_file.getvalue())
        num_pages = len(_peek)
        st.info(f"PDF detected with **{num_pages}** page(s).")
    except Exception as e:
        st.error(f"Could not read PDF: {e}")

st.markdown("**Page → Tags table** (add one row per page you want, e.g., page=3, Tags=Kitchen)")
default_df = pd.DataFrame([{"page": "", "Tags": ""}])
mapping_df = st.data_editor(
    default_df, num_rows="dynamic", use_container_width=True, key="page_tag_editor",
    column_config={
        "page": st.column_config.TextColumn("page", help="Page number (1-based)"),
        "Tags": st.column_config.TextColumn("Tags", help="Room name for that page"),
    }
)

left, right = st.columns(2)
with left:
    only_listed = st.checkbox("Only extract the pages listed above", value=True)
with right:
    pad_px = st.slider("Capture padding (pixels around link box)", 0, 12, 4, 1)
band_px = st.slider("Nearby band fallback (pixels)", 0, 40, 18, 2)

run_btn = st.button("Extract", type="primary", disabled=(pdf_file is None))

if run_btn and pdf_file:
    # Build page→Tags map
    page_to_tag = {}
    if num_pages is None:
        try:
            num_pages = len(fitz.open("pdf", pdf_file.getvalue()))
        except:
            num_pages = None
    for _, row in mapping_df.iterrows():
        p_raw = str(row.get("page","")).strip()
        t_raw = str(row.get("Tags","")).strip()
        if not p_raw:
            continue
        try:
            p_no = int(p_raw)
            if p_no >= 1 and (num_pages is None or p_no <= num_pages):
                page_to_tag[p_no] = t_raw
        except:
            continue

    pdf_bytes = pdf_file.read()
    with st.spinner("Extracting links & exact titles…"):
        df = extract_links_by_pages(
            pdf_bytes, page_to_tag,
            only_listed_pages=only_listed,
            pad_px=pad_px, band_px=band_px
        )

    if df.empty:
        st.warning("No links found. Make sure each item line is its own text box and the link covers the full line.")
    else:
        st.success(f"Extracted {len(df)} row(s).")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="canva_links_clean.csv",
            mime="text/csv"
        )

# ============== Tips for best results (shown once) ==============
with st.expander("Tips for perfect extraction"):
    st.markdown("""
- **One text box per item line.** Apply the hyperlink to the whole box.
- **No overlaps** between item boxes; add 6–10 px spacing if needed.
- Keep the **leading number** (e.g., `1.`) inside the same box as the rest of the line.
- Use consistent separators: `Type | QTY: N | FINISH: … | SIZE: …`.
- Export as **PDF Standard** with links preserved.
    """)
