# app.py — Canva PDF → clean rows (page/tags/position/type/qty/finish/size/link_url/link_text)
# plus Tab 2 (optional) to enrich URLs later if you want to keep it.

import os, re, json, time, requests
from typing import Optional, Tuple
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ============== basic setup ==============
st.set_page_config(page_title="Spec Link Toolkit", layout="wide")
st.title("🧰 Spec Link Toolkit")

UA = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
PRICE_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")

def requests_get(url: str, timeout: int=20, retries: int=2) -> Optional[requests.Response]:
    for _ in range(retries+1):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if 200 <= r.status_code < 300: return r
        except Exception:
            pass
        time.sleep(0.25)
    return None

class Timer:
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self,*exc): self.dt = time.perf_counter()-self.t0

# ============== parsing helpers (Type/QTY/Finish/Size) ==============
QTY_RE     = re.compile(r"^(?:QTY|Qty|qty|Quantity)\s*:\s*([0-9Xx]+)\s*$")
FINISH_RE  = re.compile(r"^(?:FINISH|Finish)\s*:\s*(.+)$")
SIZE_RE    = re.compile(r"^(?:SIZE|Size|Dimensions?)\s*:\s*(.+)$")
TYPE_RE    = re.compile(r"^(?:TYPE|Type)\s*:\s*(.+)$")

def parse_link_title_fields(link_text: str) -> dict:
    """Parse fields from a 'TYPE | QTY: 2 | FINISH: ... | SIZE: ...' style line."""
    fields = {"Type":"", "QTY":"", "Finish":"", "Size":""}
    s = (link_text or "").replace("\n", " | ").strip()
    if not s: return fields

    parts = []
    for chunk in s.split("|"):
        subs = [x.strip() for x in chunk.split(";")]
        parts.extend([x for x in subs if x])

    for tok in parts:
        m = TYPE_RE.match(tok)
        if m and not fields["Type"]: fields["Type"] = m.group(1).strip(); continue
        m = QTY_RE.match(tok)
        if m and not fields["QTY"]:
            q = m.group(1).strip()
            fields["QTY"] = "" if q.upper()=="XX" else q
            continue
        m = FINISH_RE.match(tok)
        if m and not fields["Finish"]: fields["Finish"] = m.group(1).strip(); continue
        m = SIZE_RE.match(tok)
        if m and not fields["Size"]: fields["Size"] = m.group(1).strip(); continue

    # Fallback Type = first unlabeled token
    if not fields["Type"]:
        for tok in parts:
            if ":" not in tok and tok:
                fields["Type"] = tok.strip(); break

    if fields["Size"]:
        fields["Size"] = re.sub(r"\s*[xX]\s*", " x ", fields["Size"]).strip()

    return fields

# ============== strict per-link title capture ==============
# Position at start (1./2./3. or with unicode dot)
POS_AT_START = re.compile(r'^\s*(\d{1,3})[.\u2024\u00B7]?\s*')
# Next bullet inside the same string (to trim accidental merges)
NEXT_BULLET  = re.compile(r'\s(\d{1,3})[.\u2024\u00B7](?=\s|[A-Z])')

def _norm_spaces(s: str) -> str:
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\s*;\s*", "; ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def extract_link_title_strict(page, rect, pad_px: float=2.0, band_px: float=18.0) -> str:
    """
    Collect only the words whose CENTER lies inside the link rectangle (±pad_px).
    If empty, look in a thin horizontal band around the rect (±band_px in Y, overlapping X).
    Uses block->line->span order for stable reading order.
    """
    if not rect: return ""
    r = fitz.Rect(rect).normalize()
    R = fitz.Rect(r.x0 - pad_px, r.y0 - pad_px, r.x1 + pad_px, r.y1 + pad_px)

    raw = page.get_text("rawdict") or {}
    words = []
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:  # text
            continue
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                tx  = span.get("text", "")
                box = span.get("bbox")
                if not tx or not box: continue
                x0,y0,x1,y1 = box
                cx = (x0+x1)/2.0; cy = (y0+y1)/2.0
                words.append((cx,cy,x0,y0,x1,y1,tx))

    kept = [(y0, x0, tx) for cx,cy,x0,y0,x1,y1,tx in words
            if fitz.Rect(R).contains(fitz.Point(cx,cy)) and tx]

    if not kept:
        band = fitz.Rect(r.x0, r.y0 - band_px, r.x1, r.y1 + band_px)
        kept = []
        for cx,cy,x0,y0,x1,y1,tx in words:
            if not tx: continue
            cy_in = (band.y0 <= cy <= band.y1)
            x_over = not (x1 < r.x0 or x0 > r.x1)
            if cy_in and x_over:
                kept.append((y0, x0, tx))

    if not kept: return ""
    kept.sort(key=lambda t: (round(t[0],3), t[1]))
    return _norm_spaces(" ".join(t[2] for t in kept))

def split_position_and_title_start(raw: str) -> tuple[str,str]:
    """Return (Position, Title). If we see the next bullet inside the same string, trim at it."""
    s = (raw or "").strip()
    if not s: return "", ""
    m = POS_AT_START.match(s)
    if not m: return "", s
    pos  = m.group(1)
    rest = s[m.end():].strip()
    m2 = NEXT_BULLET.search(rest)
    if m2:
        rest = rest[:m2.start()].strip()
    return pos, rest

# ============== main extractor: ALL links on chosen pages ==============
def extract_links_by_pages(
    pdf_bytes: bytes,
    page_to_tag: dict[int,str] | None,
    only_listed_pages: bool = True,
    pad_px: float = 2.0,
    band_px: float = 18.0,
    require_leading_number: bool = False
) -> pd.DataFrame:
    """
    - iterates every PDF link on chosen pages
    - captures per-link text (strict), splits Position, parses fields
    - writes FULL link_url (no normalization)
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
            if not uri.lower().startswith(("http://","https://")):
                continue
            rect = lnk.get("from")
            raw_title = extract_link_title_strict(page, rect, pad_px=pad_px, band_px=band_px)
            position, title = split_position_and_title_start(raw_title)
            if require_leading_number and not position:
                continue
            fields = parse_link_title_fields(title)

            rows.append({
                "page": pidx,
                "Tags": tag_value,
                "Position": position,
                "Type": fields.get("Type",""),
                "QTY": fields.get("QTY",""),
                "Finish": fields.get("Finish",""),
                "Size": fields.get("Size",""),
                "link_url": uri,     # FULL URL
                "link_text": title,  # exact label for this link (trimmed at next bullet)
            })

    return pd.DataFrame(rows)

# ============== (Optional) Enricher tab (kept simple) ==============
def pick_image_and_price_bs4(html: str, base_url: str) -> Tuple[str,str]:
    soup = BeautifulSoup(html or "", "lxml")
    img = ""
    for sel in [("meta", {"property":"og:image"}), ("meta", {"name":"og:image"}),
                ("meta", {"name":"twitter:image"}), ("meta", {"property":"twitter:image"})]:
        tag = soup.find(*sel)
        if tag and tag.get("content"):
            img = urljoin(base_url, tag["content"]); break
    if not img:
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(tag.string or "")
                objs = data if isinstance(data, list) else [data]
                for obj in objs:
                    t = obj.get("@type")
                    if t == "Product" or (isinstance(t, list) and "Product" in t):
                        im = obj.get("image")
                        if isinstance(im, list) and im:
                            img = urljoin(base_url, im[0]); break
                        if isinstance(im, str) and im:
                            img = urljoin(base_url, im); break
                if img: break
            except Exception:
                pass
    if not img:
        anyimg = soup.find("img", src=True)
        if anyimg: img = urljoin(base_url, anyimg["src"])

    price = ""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            objs = data if isinstance(data, list) else [data]
            for obj in objs:
                t = obj.get("@type")
                if t == "Product" or (isinstance(t, list) and "Product" in t):
                    offers = obj.get("offers") or {}
                    if isinstance(offers, list): offers = offers[0] if offers else {}
                    val = offers.get("price") or (offers.get("priceSpecification") or {}).get("price")
                    if val:
                        price = f"${val}" if not str(val).startswith("$") else str(val)
                        break
            if price: break
        except Exception:
            pass
    if not price:
        meta_price = soup.find("meta", attrs={"itemprop":"price"}) or soup.find("meta", attrs={"property":"product:price:amount"})
        if meta_price and meta_price.get("content"):
            val = meta_price["content"]
            price = f"${val}" if not str(val).startswith("$") else str(val)
    if not price:
        m = PRICE_RE.search(soup.get_text(" ", strip=True))
        if m: price = m.group(0)
    return img or "", price or ""

def enrich_urls(df: pd.DataFrame, url_col: str) -> pd.DataFrame:
    out = df.copy()
    if url_col not in out.columns:
        if len(out.columns) >= 2: url_col = out.columns[1]
        else:
            st.error(f"URL column '{url_col}' not found."); return out

    urls = out[url_col].astype(str).fillna("").tolist()
    imgs   = out.get("scraped_image_url", pd.Series([""]*len(out))).astype(str).tolist()
    prices = out.get("price",               pd.Series([""]*len(out))).astype(str).tolist()

    prog = st.progress(0); box = st.empty()
    t0 = time.perf_counter(); times = []

    for i, u in enumerate(urls, start=1):
        img = price = ""
        with Timer() as tt:
            r = requests_get(u)
            if r and r.text:
                img, price = pick_image_and_price_bs4(r.text, u)
        imgs[i-1] = img; prices[i-1] = price
        times.append(tt.dt)
        avg = sum(times)/len(times); eta = int((len(urls)-i)*avg)
        box.write(f"Processed {i}/{len(urls)} • last {tt.dt:.2f}s • avg {avg:.2f}s/link • ETA ~{eta}s")
        prog.progress(i/len(urls)); time.sleep(0.1)

    out["scraped_image_url"] = imgs
    out["price"] = prices
    return out

# ============== UI ==============
tab1, tab2 = st.tabs([
    "1) Extract from PDF (pages + Tags + Position + full titles)",
    "2) (Optional) Enrich CSV (Image URL + Price)"
])

with tab1:
    st.caption("Map **page → Tags**, then extract **ALL links** on those pages. Each link stays on its own row, with Position and parsed fields.")
    pdf_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_extractor")

    num_pages = None
    if pdf_file:
        try:
            _peek = fitz.open("pdf", pdf_file.getvalue())
            num_pages = len(_peek)
            st.info(f"PDF detected with **{num_pages}** page(s).")
        except Exception as e:
            st.error(f"Could not read PDF: {e}")

    st.markdown("**Page → Tags table**")
    default_df = pd.DataFrame([{"page":"", "Tags":""}])
    mapping_df = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="page_tag_editor",
        column_config={
            "page": st.column_config.TextColumn("page", help="Page number (1-based)"),
            "Tags": st.column_config.TextColumn("Tags", help="Room name for that page (goes to CSV column 'Tags')"),
        }
    )

    st.markdown("**Capture settings (usually fine to leave as-is)**")
    colA, colB, colC = st.columns(3)
    with colA:
        only_listed = st.checkbox("Only extract pages listed above", value=True)
    with colB:
        pad_px = st.slider("Rect pad (pixels)", 0, 10, 2, 1)
    with colC:
        band_px = st.slider("Nearby band height (pixels)", 0, 40, 18, 2)

    require_num = st.checkbox("Require a leading number (Position) on each item", value=False)

    run1 = st.button("Extract", type="primary", disabled=(pdf_file is None), key="extract_btn")
    if run1 and pdf_file:
        # build page→tag dict
        page_to_tag = {}
        if num_pages is None:
            try: num_pages = len(fitz.open("pdf", pdf_file.getvalue()))
            except: pass
        for _, row in mapping_df.iterrows():
            p_raw = str(row.get("page","")).strip()
            t_raw = str(row.get("Tags","")).strip()
            if not p_raw: continue
            try:
                p_no = int(p_raw)
                if p_no >= 1 and (num_pages is None or p_no <= num_pages):
                    page_to_tag[p_no] = t_raw
            except: continue

        pdf_bytes = pdf_file.read()
        with st.spinner("Extracting links, positions & titles…"):
            df = extract_links_by_pages(
                pdf_bytes, page_to_tag,
                only_listed_pages=only_listed,
                pad_px=pad_px, band_px=band_px,
                require_leading_number=require_num
            )
        if df.empty:
            st.info("No links found. Make sure your Canva file exports *live hyperlinks* (not flattened).")
        else:
            st.success(f"Extracted {len(df)} row(s).")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="canva_links_with_position.csv",
                mime="text/csv"
            )

with tab2:
    st.caption("Optional helper to fetch Image URL + Price from a CSV of product URLs (no Firecrawl here).")
    csv_file = st.file_uploader("Upload links CSV", type=["csv"], key="csv_uploader")
    if csv_file is not None:
        try:
            df_in = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_in = None

        if df_in is not None:
            st.write("Preview:", df_in.head())
            url_col_guess = "Product URL" if "Product URL" in df_in.columns else df_in.columns[min(1, len(df_in)-1)]
            url_col = st.text_input("URL column name", url_col_guess)
            if st.button("Enrich (Image URL + Price)", key="enrich_btn"):
                with st.spinner("Scraping image + price..."):
                    df_out = enrich_urls(df_in, url_col)
                st.success("Enriched! ✅")
                st.dataframe(df_out, use_container_width=True)
                out_csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download enriched CSV",
                    data=out_csv,
                    file_name="links_enriched.csv",
                    mime="text/csv",
                )
