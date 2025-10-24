# app.py — Canva PDF extractor:
# - Choose pages via an editable table: page | Tags (room name)
# - One row per link on those pages
# - Parse Type/QTY/Finish/Size FROM THE LINK TITLE TEXT

import re
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from urllib.parse import urlparse, urlunparse

# ========================= Helpers =========================
URL_RE = re.compile(r'https?://\S+', re.IGNORECASE)
QTY_RE     = re.compile(r"^(?:QTY|Qty|qty)\s*:\s*(\d+)\s*$")
FINISH_RE  = re.compile(r"^(?:Finish|FINISH)\s*:\s*(.+)$")
SIZE_RE    = re.compile(r"^(?:Size|SIZE)\s*:\s*(.+)$")
TYPE_RE    = re.compile(r"^(?:Type|TYPE)\s*:\s*(.+)$")

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

def parse_link_title_fields(link_text: str) -> dict:
    """
    Parse a link title string into {Type, QTY, Finish, Size}.
    Splits on '|' and ';', then matches key:value pairs (case-insensitive).
    If no explicit 'Type:' is present, the first standalone token becomes Type.
    """
    fields = {"Type": "", "QTY": "", "Finish": "", "Size": ""}

    if not link_text:
        return fields

    # Normalize newlines to ' | ' so all separators behave the same.
    s = (link_text or "").replace("\n", " | ")
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
                # Avoid tokens that are just labels
                if tok.lower() not in ("finish", "size", "qty", "quantity"):
                    fields["Type"] = tok.strip()
                    break

    return fields

def extract_links_by_pages(
    pdf_bytes: bytes,
    page_to_tag: dict[int, str] | None,
    only_listed_pages: bool = True
) -> pd.DataFrame:
    """
    Extract links. If page_to_tag is given, and only_listed_pages=True,
    process ONLY those pages and use its Tags value.
    If page_to_tag is empty or only_listed_pages=False, process all pages
    and apply Tags when provided for that page (else blank).

    For each link, parse Type/QTY/Finish/Size from the LINK TITLE TEXT.
    """
    doc = fitz.open("pdf", pdf_bytes)
    rows = []

    listed_pages = set(page_to_tag.keys()) if page_to_tag else set()

    for pidx, page in enumerate(doc, start=1):
        if only_listed_pages and page_to_tag:
            if pidx not in listed_pages:
                continue

        # Collect all links on this page
        page_links = []
        for lnk in page.get_links():
            uri = lnk.get("uri") or ""
            if uri.lower().startswith(("http://", "https://")):
                rect = lnk.get("from")
                link_title = ""
                if rect:
                    try:
                        link_title = page.get_textbox(rect).strip()
                    except:
                        link_title = ""
                page_links.append((norm_url(uri), link_title))

        if not page_links:
            continue

        tag_value = (page_to_tag or {}).get(pidx, "")  # empty if not provided

        for url, link_title in page_links:
            parsed = parse_link_title_fields(link_title)
            rows.append({
                "page": pidx,
                "Tags": tag_value,
                "Type": parsed.get("Type", ""),
                "QTY": parsed.get("QTY", ""),
                "Finish": parsed.get("Finish", ""),
                "Size": parsed.get("Size", ""),
                "link_url": url,
                "link_text": link_title,
            })

    return pd.DataFrame(rows)

# ========================= Streamlit UI =========================
st.set_page_config(page_title="Canva PDF → Links (Tags + fields from link title)", layout="wide")
st.title("📄 Canva PDF Extractor — Tags per Page + Fields from Link Title")

st.caption(
    "Upload a Canva PDF, list the pages you care about with their room **Tags**, "
    "and get one row per link. Type/QTY/Finish/Size are parsed from the link’s title text."
)

pdf_file = st.file_uploader("Upload PDF", type="pdf")

num_pages = None
if pdf_file:
    try:
        _peek = fitz.open("pdf", pdf_file.getvalue())
        num_pages = len(_peek)
        st.info(f"PDF detected with **{num_pages}** page(s).")
    except Exception as e:
        st.error(f"Could not read PDF: {e}")

st.markdown("### Page → Tag mapping")
st.caption("Edit the table: put a page number (1-based) and a tag (e.g., Kitchen, Bar). Add multiple rows as needed.")

# Start with a single empty row for convenience
default_df = pd.DataFrame([{"page": "", "Tags": ""}])

# Editable table (user can add rows with the + button)
mapping_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True,
    key="page_tag_editor",
    column_config={
        "page": st.column_config.TextColumn("page", help="Page number (1-based)"),
        "Tags": st.column_config.TextColumn("Tags", help="Room name for that page"),
    }
)

col_a, col_b = st.columns([1,1])
with col_a:
    only_listed = st.checkbox(
        "Only extract pages listed above",
        value=True,
        help="If unchecked, the app will extract from ALL pages and apply Tags where provided."
    )
with col_b:
    st.caption("Tip: If you want to prefill, type 1..N in the 'page' column rows, leave Tags empty for pages you don’t care about.")

run = st.button("Run extraction", type="primary", disabled=(pdf_file is None))

if run:
    if not pdf_file:
        st.warning("Please upload a PDF first.")
    else:
        # Build mapping dict {page_number: tag}
        page_to_tag = {}
        for _, row in mapping_df.iterrows():
            p_raw = str(row.get("page", "")).strip()
            t_raw = str(row.get("Tags", "")).strip()
            if not p_raw:
                continue
            try:
                p_no = int(p_raw)
                if p_no >= 1 and (num_pages is None or p_no <= num_pages):
                    # include even if Tags is empty (so you can list which pages to extract)
                    page_to_tag[p_no] = t_raw
            except:
                continue

        pdf_bytes = pdf_file.read()

        with st.spinner("Extracting links and parsing fields from link titles…"):
            df = extract_links_by_pages(pdf_bytes, page_to_tag, only_listed_pages=only_listed)

        if df.empty:
            if only_listed and page_to_tag:
                st.info("No links found on the listed pages. Check page numbers or ensure links exist.")
            else:
                st.info("No links found. You may need to adjust your Canva export or verify the PDF contains live links.")
        else:
            st.success(f"Extracted {len(df)} link row(s).")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="canva_links_with_tags_and_fields.csv",
                mime="text/csv"
            )
