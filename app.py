# app.py — Canva PDF selective extractor with Tags + parsed bottom-left fields
import re
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from urllib.parse import urlparse, urlunparse

# ========================= Helpers =========================
URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)

def norm_url(u: str) -> str:
    """Normalize a URL: strip spaces, lowercase host, drop trailing slash in path."""
    if not isinstance(u, str) or not u.strip():
        return ""
    try:
        p = urlparse(u.strip())
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = p.path.rstrip("/")
        return urlunparse((p.scheme.lower(), netloc, path, "", "", ""))
    except Exception:
        return u.strip()

def parse_page_selection(pages_str: str, num_pages: int) -> set[int]:
    """
    Convert "1,3-5,8" into {1,3,4,5,8}. Pages are 1-based.
    Clamp to [1, num_pages]. Blank = all pages.
    """
    if not pages_str:
        return set(range(1, num_pages + 1))
    out = set()
    for chunk in pages_str.split(","):
        c = chunk.strip()
        if not c:
            continue
        if "-" in c:
            a, b = c.split("-", 1)
            try:
                start = max(1, min(num_pages, int(a)))
                end   = max(1, min(num_pages, int(b)))
                lo, hi = (start, end) if start <= end else (end, start)
                out.update(range(lo, hi + 1))
            except:
                pass
        else:
            try:
                v = max(1, min(num_pages, int(c)))
                out.add(v)
            except:
                pass
    return out or set(range(1, num_pages + 1))

def parse_page_tags_map(mapping_str: str) -> dict[int, str]:
    """
    Convert "3:Kitchen, 5:Bar, 8:Powder" → {3: "Kitchen", 5: "Bar", 8: "Powder"}.
    Ignores malformed pairs.
    """
    out = {}
    if not mapping_str:
        return out
    for pair in mapping_str.split(","):
        p = pair.strip()
        if not p or ":" not in p:
            continue
        idx, tag = p.split(":", 1)
        try:
            page_no = int(idx.strip())
            tag_val = tag.strip()
            if tag_val:
                out[page_no] = tag_val
        except:
            continue
    return out

# -------- bottom-left parser: Type / QTY / Finish / Size ----------
QTY_RE     = re.compile(r"^(?:QTY|Qty|qty)\s*:\s*(\d+)\s*$")
FINISH_RE  = re.compile(r"^(?:Finish|FINISH)\s*:\s*(.+)$")
SIZE_RE    = re.compile(r"^(?:Size|SIZE)\s*:\s*(.+)$")
TYPE_RE    = re.compile(r"^(?:Type|TYPE)\s*:\s*(.+)$")

def parse_bottom_left_fields(raw_text: str) -> dict:
    """
    Parse a bottom-left block into {Type, QTY, Finish, Size}.
    Splits on '|' first, then on ';'. Accepts tokens like:
      - "Pendant"  (becomes Type if no explicit Type:)
      - "QTY: 2"
      - "Finish: Black"
      - "Size: 12\" x 8\""
    Returns empty strings if not found.
    """
    fields = {"Type": "", "QTY": "", "Finish": "", "Size": ""}

    if not raw_text.strip():
        return fields

    # Normalize separators: treat newlines as '|'
    s = raw_text.replace("\n", " | ")
    parts = []
    for chunk in s.split("|"):
        sub = [x.strip() for x in chunk.split(";")]
        parts.extend([x for x in sub if x])

    # Pass 1: explicit key:value tokens
    for tok in parts:
        m = TYPE_RE.match(tok)
        if m and not fields["Type"]:
            fields["Type"] = m.group(1).strip()
            continue
        m = QTY_RE.match(tok)
        if m and not fields["QTY"]:
            fields["QTY"] = m.group(1).strip()
            continue
        m = FINISH_RE.match(tok)
        if m and not fields["Finish"]:
            fields["Finish"] = m.group(1).strip()
            continue
        m = SIZE_RE.match(tok)
        if m and not fields["Size"]:
            fields["Size"] = m.group(1).strip()
            continue

    # Pass 2: infer Type from first standalone token (no colon) if still empty
    if not fields["Type"]:
        for tok in parts:
            if ":" not in tok and not QTY_RE.match(tok):
                # Avoid tokens that are just labels like "Finish" or "Size"
                if tok.lower() not in ("finish", "size", "qty", "quantity"):
                    fields["Type"] = tok.strip()
                    break

    return fields

def extract_selected_pages(
    pdf_bytes: bytes,
    selected_pages: set[int],
    page_tags: dict[int, str],
    bl_left_ratio: float = 0.65,  # left 65% of page width
    bl_bottom_start: float = 0.55 # bottom area begins at 55% of page height
) -> pd.DataFrame:
    """
    For each selected page:
      - bottom-left: parse into Type/QTY/Finish/Size
      - links: every http(s) link on the page → each link = one row
      - Tags: from page_tags mapping (or blank if not provided)
    """
    doc = fitz.open("pdf", pdf_bytes)
    rows = []

    for pidx, page in enumerate(doc, start=1):
        if pidx not in selected_pages:
            continue

        w, h = page.rect.width, page.rect.height

        # --- Bottom-left “materials list” area ---
        bottom_left_rect = fitz.Rect(0, h * bl_bottom_start, w * bl_left_ratio, h)
        bl_chunks = []
        for x0, y0, x1, y1, txt, *_ in page.get_text("blocks"):
            r = fitz.Rect(x0, y0, x1, y1)
            if r.intersects(bottom_left_rect):
                t = str(txt).strip()
                if t:
                    bl_chunks.append(t)
        bottom_left_text = "\n".join(bl_chunks).strip()
        parsed = parse_bottom_left_fields(bottom_left_text)

        # --- Collect links anywhere on the page ---
        page_links = []
        for lnk in page.get_links():
            uri = lnk.get("uri") or ""
            if uri.lower().startswith(("http://", "https://")):
                rect = lnk.get("from")
                link_text = ""
                if rect:
                    try:
                        link_text = page.get_textbox(rect).strip()
                    except:
                        link_text = ""
                page_links.append((norm_url(uri), link_text))

        tag_value = page_tags.get(pidx, "")

        # One row per link (with the same parsed fields + tag for that page)
        if page_links:
            for url, ltxt in page_links:
                rows.append({
                    "page": pidx,
                    "Tags": tag_value,
                    "Type": parsed.get("Type", ""),
                    "QTY": parsed.get("QTY", ""),
                    "Finish": parsed.get("Finish", ""),
                    "Size": parsed.get("Size", ""),
                    "link_url": url,
                    "link_text": ltxt,
                })
        else:
            # If you want to include pages that had no links, uncomment:
            # rows.append({
            #     "page": pidx,
            #     "Tags": tag_value,
            #     "Type": parsed.get("Type",""),
            #     "QTY": parsed.get("QTY",""),
            #     "Finish": parsed.get("Finish",""),
            #     "Size": parsed.get("Size",""),
            #     "link_url": "",
            #     "link_text": "",
            # })
            pass

    return pd.DataFrame(rows)

# ========================= Streamlit UI =========================
st.set_page_config(page_title="Canva PDF → Links with Tags & Parsed Fields", layout="wide")
st.title("📄 Canva PDF Extractor — Selected Pages + Tags + Parsed Bottom-left (Type/QTY/Finish/Size)")

st.caption(
    "Upload a Canva-exported PDF, choose pages, and map pages to room tags. "
    "Output: one row per link with parsed bottom-left fields."
)

pdf_file = st.file_uploader("Upload PDF", type="pdf")

# Peek page count for friendlier prompts
num_pages_display = ""
if pdf_file:
    try:
        tmp_doc = fitz.open("pdf", pdf_file.getvalue())
        num_pages_display = f"(PDF has {len(tmp_doc)} pages)"
    except Exception:
        num_pages_display = ""

col1, col2 = st.columns([1, 1])
with col1:
    pages_input = st.text_input(
        f"Pages to extract {num_pages_display} (e.g. 1,3-5,8). Leave blank for ALL pages.",
        value=""
    )
with col2:
    tags_input = st.text_input(
        "Page→Tag map (e.g. 3:Kitchen, 5:Bar, 8:Powder).",
        value=""
    )

with st.expander("Advanced (optional): tweak bottom-left region heuristics"):
    bl_left_ratio = st.slider("Bottom-left width ratio (0.40–0.90)", 0.40, 0.90, 0.65, 0.01)
    bl_bottom_start = st.slider("Bottom area starts at page height % (0.40–0.80)", 0.40, 0.80, 0.55, 0.01)

run_btn = st.button("Run selective extract", type="primary", disabled=(pdf_file is None))

if run_btn:
    if not pdf_file:
        st.warning("Please upload a PDF first.")
    else:
        pdf_bytes = pdf_file.read()
        doc = fitz.open("pdf", pdf_bytes)
        selected_pages = parse_page_selection(pages_input, num_pages=len(doc))
        page_tags = parse_page_tags_map(tags_input)

        # Warn if user mapped tags for pages they didn't select
        extra_tagged = sorted(set(page_tags.keys()) - selected_pages)
        if extra_tagged:
            st.info(f"Note: you provided tags for pages not selected: {extra_tagged}")

        with st.spinner("Extracting…"):
            df = extract_selected_pages(
                pdf_bytes,
                selected_pages,
                page_tags,
                bl_left_ratio=bl_left_ratio,
                bl_bottom_start=bl_bottom_start
            )

        if df.empty:
            st.info("No links found on the selected pages. Try different pages or adjust the bottom-left region.")
        else:
            st.success(f"Extracted {len(df)} row(s) from pages: {sorted(selected_pages)}")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="canva_selected_pages_extract.csv",
                mime="text/csv"
            )
