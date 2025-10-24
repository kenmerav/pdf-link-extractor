# app.py — Canva PDF selective extractor (pages + page→tag mapping)
import re
import io
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

def extract_selected_pages(
    pdf_bytes: bytes,
    selected_pages: set[int],
    page_tags: dict[int, str],
    bl_left_ratio: float = 0.65,  # left 65% of page width
    bl_bottom_start: float = 0.55 # bottom area begins at 55% of page height
) -> pd.DataFrame:
    """
    For each selected page:
      - bottom-left: capture the full “materials list” block as one string
      - links: every http(s) link on the page → each link = one row
      - Tags: from page_tags mapping (or blank if not provided)

    Region heuristics (tuned for Canva spec pages):
      - Bottom-left rectangle = x:[0 .. w*0.65], y:[h*0.55 .. h]
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

        # One row per link (with the same bottom-left + tag for that page)
        if page_links:
            for url, ltxt in page_links:
                rows.append({
                    "page": pidx,
                    "Tags": tag_value,
                    "bottom_left_text": bottom_left_text,
                    "link_url": url,
                    "link_text": ltxt,
                })
        else:
            # If you also want to include pages that had no links, uncomment below:
            # rows.append({
            #     "page": pidx,
            #     "Tags": tag_value,
            #     "bottom_left_text": bottom_left_text,
            #     "link_url": "",
            #     "link_text": "",
            # })
            pass

    return pd.DataFrame(rows)

# ========================= Streamlit UI =========================
st.set_page_config(page_title="Canva PDF → Links (selected pages + tags)", layout="wide")
st.title("📄 Canva PDF Extractor — Selected Pages + Tags")

st.caption(
    "Upload a Canva-exported PDF, choose which pages to process, and map pages to room tags. "
    "Output: one row per link with the page’s full bottom-left text."
)

pdf_file = st.file_uploader("Upload PDF", type="pdf")

# Peek page count for friendlier prompts
num_pages_display = ""
num_pages_val = None
if pdf_file:
    try:
        tmp_doc = fitz.open("pdf", pdf_file.getvalue())
        num_pages_val = len(tmp_doc)
        num_pages_display = f"(PDF has {num_pages_val} pages)"
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
        "Page→Tag map (e.g. 3:Kitchen, 5:Bar, 8:Powder). Leave blank if you don't want tags.",
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
