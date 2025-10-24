# app.py — Canva PDF → clean rows with Position, Type, QTY, Finish, Size, full URL & link_text

import os, re, time, json, requests
from typing import Optional, Tuple, List, Dict
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd

# ---------------- UI setup ----------------
st.set_page_config(page_title="Spec Link Toolkit", layout="wide")
st.title("🧰 Spec Link Toolkit — Canva PDF → Clean Links")

# --------------- Parsers ------------------
POS_AT_START = re.compile(r'^\s*(\d{1,3})[.\u2024\u00B7]?\s*')
NEXT_BULLET  = re.compile(r'\s(\d{1,3})[.\u2024\u00B7](?=\s|[A-Z])')

QTY_RE     = re.compile(r"^(?:QTY|Qty|qty|Quantity)\s*:\s*([0-9Xx]+)\s*$")
FINISH_RE  = re.compile(r"^(?:FINISH|Finish)\s*:\s*(.+)$")
SIZE_RE    = re.compile(r"^(?:SIZE|Size|Dimensions?)\s*:\s*(.+)$")
TYPE_RE    = re.compile(r"^(?:TYPE|Type)\s*:\s*(.+)$")

def norm_spaces(s: str) -> str:
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\s*;\s*", "; ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def split_position_and_title(raw: str) -> Tuple[str, str]:
    """Pull Position at the start; trim if another bullet appears later."""
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

def parse_fields_from_title(link_text: str) -> Dict[str, str]:
    """Parse Type / QTY / Finish / Size from a single-line title."""
    fields = {"Type":"", "QTY":"", "Finish":"", "Size":""}
    s = norm_spaces((link_text or "").replace("\n", " | "))
    if not s:
        return fields

    parts: List[str] = []
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
            fields["QTY"] = "" if q.upper()=="XX" else q
            continue
        m = FINISH_RE.match(tok)
        if m and not fields["Finish"]:
            fields["Finish"] = m.group(1).strip(); continue
        m = SIZE_RE.match(tok)
        if m and not fields["Size"]:
            fields["Size"] = m.group(1).strip(); continue

    if not fields["Type"]:
        # Fallback: first unlabeled token
        for tok in parts:
            if ":" not in tok and tok:
                fields["Type"] = tok.strip()
                break

    if fields["Size"]:
        fields["Size"] = re.sub(r"\s*[xX]\s*", " x ", fields["Size"]).strip()

    return fields

# --------------- Title capture (robust) ------------------
def words_from_page(page) -> List[tuple]:
    """
    Return words with geometry in stable reading order.
    PyMuPDF 'words' = [x0, y0, x1, y1, "word", block_no, line_no, word_no]
    """
    words = page.get_text("words") or []
    # sort by block, then line, then word
    words.sort(key=lambda w: (w[5], w[6], w[7]))
    return words

def capture_title_for_link(page, link_rect, pad_px: float, band_px: float) -> str:
    """
    Keep only words whose CENTER lies inside the link rect (±pad_px).
    If none, look in a thin horizontal band around the rect (±band_px in Y, overlapping X).
    """
    if not link_rect:
        return ""
    r = fitz.Rect(link_rect).normalize()
    R = fitz.Rect(r.x0 - pad_px, r.y0 - pad_px, r.x1 + pad_px, r.y1 + pad_px)

    words = words_from_page(page)
    kept: List[tuple] = []
    for x0,y0,x1,y1,txt,blk,lin,wn in words:
        if not txt:
            continue
        cx = (x0+x1)/2.0; cy = (y0+y1)/2.0
        if R.contains(fitz.Point(cx, cy)):
            kept.append((blk, lin, wn, txt))

    if not kept:
        # Nearby band fallback
        band = fitz.Rect(r.x0, r.y0 - band_px, r.x1, r.y1 + band_px)
        for x0,y0,x1,y1,txt,blk,lin,wn in words:
            if not txt:
                continue
            cx = (x0+x1)/2.0; cy = (y0+y1)/2.0
            cy_in = (band.y0 <= cy <= band.y1)
            x_over = not (x1 < r.x0 or x0 > r.x1)
            if cy_in and x_over:
                kept.append((blk, lin, wn, txt))

    if not kept:
        return ""

    kept.sort(key=lambda t: (t[0], t[1], t[2]))
    text = " ".join(k[3] for k in kept)
    return norm_spaces(text)

# --------------- Main extraction ------------------
def extract_links_by_pages(pdf_bytes: bytes,
                           page_to_tag: Dict[int, str] | None,
                           only_listed_pages: bool = True,
                           pad_px: float = 2.0,
                           band_px: float = 14.0,
                           require_leading_number: bool = False) -> pd.DataFrame:
    """
    - Iterate all hyperlinks on the chosen pages
    - Capture **per-link** title (no merging)
    - Split Position + parse fields
    - Keep FULL link URL
    """
    doc = fitz.open("pdf", pdf_bytes)
    listed = set(page_to_tag.keys()) if page_to_tag else set()
    rows = []

    for pidx, page in enumerate(doc, start=1):
        if only_listed_pages and page_to_tag and pidx not in listed:
            continue

        tag_value = (page_to_tag or {}).get(pidx, "")

        for lnk in page.get_links():
            url = (lnk.get("uri") or "").strip()
            if not url.lower().startswith(("http://", "https://")):
                continue

            rect = lnk.get("from")
            raw = capture_title_for_link(page, rect, pad_px=pad_px, band_px=band_px)
            position, title = split_position_and_title(raw)

            if require_leading_number and not position:
                # If you toggle this on, skip unnumbered items
                continue

            fields = parse_fields_from_title(title)

            rows.append({
                "page": pidx,
                "Tags": tag_value,
                "Position": position,
                "Type": fields.get("Type",""),
                "QTY": fields.get("QTY",""),
                "Finish": fields.get("Finish",""),
                "Size": fields.get("Size",""),
                "link_url": url,      # FULL original URL
                "link_text": title,   # exact label for this link
            })

    return pd.DataFrame(rows)

# --------------- UI ------------------
tab1 = st.container()
with tab1:
    st.caption("Map **page → Tags**, then extract **ALL links** on those pages. Each link stays in its own row. Titles come from the text that sits on/near the link box.")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")

    num_pages = None
    if pdf_file:
        try:
            _peek = fitz.open("pdf", pdf_file.getvalue())
            num_pages = len(_peek)
            st.info(f"PDF detected with **{num_pages}** page(s).")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")

    st.markdown("**Page → Tags table**")
    default_df = pd.DataFrame([{"page": "", "Tags": ""}])
    mapping_df = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "page": st.column_config.TextColumn("page", help="Page number (1-based)"),
            "Tags": st.column_config.TextColumn("Tags", help="Room name for that page (goes to CSV column 'Tags')"),
        }
    )

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        only_listed = st.checkbox("Only extract listed pages", value=True)
    with col2:
        pad_px = st.slider("Rect pad (px)", 0, 10, 2, 1)
    with col3:
        band_px = st.slider("Nearby band (px)", 0, 40, 14, 2)
    with col4:
        require_num = st.checkbox("Require leading number (Position)", value=False)

    if st.button("Extract", type="primary", disabled=(pdf_file is None)):
        # build mapping
        page_to_tag = {}
        if num_pages is None:
            try:
                num_pages = len(fitz.open("pdf", pdf_file.getvalue()))
            except:
                pass
        for _, row in mapping_df.iterrows():
            p_raw = str(row.get("page","")).strip()
            t_raw = str(row.get("Tags","")).strip()
            if not p_raw: continue
            try:
                p_no = int(p_raw)
                if p_no >= 1 and (num_pages is None or p_no <= num_pages):
                    page_to_tag[p_no] = t_raw
            except:
                continue

        pdf_bytes = pdf_file.read()
        with st.spinner("Extracting links and titles…"):
            df = extract_links_by_pages(
                pdf_bytes, page_to_tag,
                only_listed_pages=only_listed,
                pad_px=pad_px, band_px=band_px,
                require_leading_number=require_num
            )

        if df.empty:
            st.warning("No links found. Make sure each list line is its own text box AND the hyperlink rectangle overlaps that text.")
        else:
            st.success(f"Extracted {len(df)} row(s).")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="canva_links_clean.csv",
                mime="text/csv"
            )
