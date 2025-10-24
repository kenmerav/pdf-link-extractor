# app.py — Full app with:
# 1) Canva PDF → Page→Tags + strict link title capture + Position + Type/QTY/Finish/Size parsing
# 2) Enrich CSV (Firecrawl v2 → bs4 fallback) to get image URL + price
# 3) Test a single URL

import os, re, json, time, requests
from typing import Optional, Tuple
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse, urljoin

# ========================= Shared helpers =========================
st.set_page_config(page_title="Spec Link Toolkit", layout="wide")
st.title("🧰 Spec Link Toolkit")

PRICE_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")  # $1,234.56
UA = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def norm_url(u: str) -> str:
    """Keep scheme/host/path/params/query exactly; only strip fragment. Lowercase host."""
    if not isinstance(u, str) or not u.strip():
        return ""
    try:
        p = urlparse(u.strip())
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return urlunparse((p.scheme, netloc, p.path, p.params, p.query, ""))  # keep query intact, drop fragment
    except Exception:
        return u.strip()

def requests_get(url: str, timeout: int = 20, retries: int = 2) -> Optional[requests.Response]:
    for _ in range(retries+1):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if 200 <= r.status_code < 300:
                return r
        except Exception:
            pass
        time.sleep(0.25)
    return None

class Timer:
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *exc): self.dt = time.perf_counter() - self.t0

# ========================= TAB 1: Canva PDF extractor =========================
# Robust regexes for parsing the link title into columns
QTY_RE     = re.compile(r"^(?:QTY|Qty|qty|Quantity)\s*:\s*([0-9Xx]+)\s*$")
FINISH_RE  = re.compile(r"^(?:FINISH|Finish)\s*:\s*(.+)$")
SIZE_RE    = re.compile(r"^(?:SIZE|Size|Dimensions?)\s*:\s*(.+)$")
TYPE_RE    = re.compile(r"^(?:TYPE|Type)\s*:\s*(.+)$")

def parse_link_title_fields(link_text: str) -> dict:
    """
    Parse 'PENDANT | QTY: 1 | FINISH: Brass, White | SIZE: 48"' style strings.
    Accepts pipes or semicolons. Infers Type from first unlabeled token if needed.
    """
    fields = {"Type": "", "QTY": "", "Finish": "", "Size": ""}

    s = (link_text or "").replace("\n", " | ").strip()
    if not s:
        return fields

    # split on pipes first, then on semicolons
    parts = []
    for chunk in s.split("|"):
        subs = [x.strip() for x in chunk.split(";")]
        parts.extend([x for x in subs if x])

    # explicit labeled tokens
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
            fields["Finish"] = m.group(1).strip()
            continue

        m = SIZE_RE.match(tok)
        if m and not fields["Size"]:
            fields["Size"] = m.group(1).strip()
            continue

    # infer Type from first unlabeled token
    if not fields["Type"]:
        for tok in parts:
            if ":" in tok:
                continue
            if tok.strip():
                fields["Type"] = tok.strip()
                break

    if fields["Size"]:
        # normalize spacing around X
        fields["Size"] = re.sub(r"\s*[xX]\s*", " x ", fields["Size"]).strip()

    return fields

# Numbered-bullet detection and cleaning (Position + single-item title)
# was: NUM_DOT = re.compile(r"\b(\d{1,3})\.\s")
# Replace previous NUM_DOT with this more tolerant version
# Matches: 1. PENDANTS  |  1.PENDANTS  |  1․PENDANTS (Unicode dot)
NUM_BULLET = re.compile(r'(?<!\d)(\d{1,3})[.\u2024\u00B7](?:\s|$)')

def split_position_and_title(raw: str) -> tuple[str, str]:
    """
    Return (position, clean_title) for ONE numbered item:
      - Start at the first bullet "N." (space optional; Unicode dot allowed)
      - Stop just BEFORE the next bullet
      - Strip the leading "N." itself from the returned title
    """
    s = (raw or "").strip()
    if not s:
        return "", ""

    m1 = NUM_BULLET.search(s)
    if not m1:
        # No bullet found; normalize separators and return as-is
        clean = re.sub(r"\s*\|\s*", " | ", s)
        clean = re.sub(r"\s*;\s*", "; ", clean)
        clean = re.sub(r"\s{2,}", " ", clean).strip()
        return "", clean

    pos_num = m1.group(1)
    start = m1.end()

    # Find the next bullet AFTER the first one
    m2 = NUM_BULLET.search(s, pos=start)
    end = m2.start() if m2 else len(s)

    # Slice current item only
    clean = s[start:end].strip()

    # Normalize separators/spaces (keeps `32"` intact)
    clean = re.sub(r"\s*\|\s*", " | ", clean)
    clean = re.sub(r"\s*;\s*", "; ", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()

    return pos_num, clean


def extract_link_title_exact(page, rect, pad_px: float = 4.0, inflate_pct: float = 0.02, overlap_min: float = 0.10) -> str:
    """
    Collect ALL words that overlap the link rectangle by at least `overlap_min` (10% default).
    Inflate the rect a little to avoid edge truncation. Sort top→down, left→right.
    """
    if not rect:
        return ""
    r = fitz.Rect(rect).normalize()

    # Inflate by percent of page + a few pixels
    W, H = page.rect.width, page.rect.height
    dx = max(pad_px, W * inflate_pct)
    dy = max(pad_px, H * inflate_pct)
    R = fitz.Rect(r.x0 - dx, r.y0 - dy, r.x1 + dx, r.y1 + dy)

    def intersect_area(a: fitz.Rect, b: fitz.Rect) -> float:
        x0 = max(a.x0, b.x0); y0 = max(a.y0, b.y0)
        x1 = min(a.x1, b.x1); y1 = min(a.y1, b.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return (x1 - x0) * (y1 - y0)

    words = page.get_text("words")  # [x0,y0,x1,y1, text, block, line, word]
    kept = []
    for x0, y0, x1, y1, wtext, *_ in words:
        wb = fitz.Rect(x0, y0, x1, y1)
        inter = intersect_area(R, wb)
        if inter <= 0:
            continue
        # manual area (older PyMuPDF doesn’t have wb.get_area())
        word_area = max((wb.x1 - wb.x0) * (wb.y1 - wb.y0), 1.0)
        if inter / word_area >= overlap_min:
            kept.append((y0, x0, wtext))

    if not kept:
        return ""

    kept.sort(key=lambda t: (round(t[0], 2), t[1]))
    txt = " ".join(t[2] for t in kept)

    # Normalize separators: pipes & semicolons, collapse extra spaces
    txt = re.sub(r"\s*\|\s*", " | ", txt)
    txt = re.sub(r"\s*;\s*", "; ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def extract_links_by_pages(
    pdf_bytes: bytes,
    page_to_tag: dict[int, str] | None,
    only_listed_pages: bool = True
) -> pd.DataFrame:
    """
    Collect *all* link annotations on selected pages (or all pages).
    - FULL link (including query) via norm_url()
    - Title = exact text overlapping the link’s rectangle
    - Enforce per-item boundaries using leading number "N. "
    - Output Position (the N) and parsed Type/QTY/Finish/Size
    """
    doc = fitz.open("pdf", pdf_bytes)
    rows = []
    listed = set(page_to_tag.keys()) if page_to_tag else set()

    for pidx, page in enumerate(doc, start=1):
        if only_listed_pages and page_to_tag and pidx not in listed:
            continue

        tag_value = (page_to_tag or {}).get(pidx, "")

        for lnk in page.get_links():
            uri = (lnk.get("uri") or "").strip()      # keep FULL URL
            rect = lnk.get("from")
            raw_title = extract_link_title_exact(page, rect, pad_px=4.0, inflate_pct=0.02, overlap_min=0.10)

            # Position + single-item title
            position, title = split_position_and_title(raw_title)

            # Parse fields from the cleaned title
            parsed = parse_link_title_fields(title)

            rows.append({
                "page": pidx,
                "Tags": tag_value,
                "Position": position,               # new column
                "Type": parsed.get("Type",""),
                "QTY": parsed.get("QTY",""),
                "Finish": parsed.get("Finish",""),
                "Size": parsed.get("Size",""),
                "link_url": norm_url(uri),          # FULL URL (keeps query)
                "link_text": title,                 # cleaned, single-item title
            })

    return pd.DataFrame(rows)

# ========================= TAB 2/3: Firecrawl + fallback =========================
def firecrawl_scrape_v2(url: str, api_key: str, mode: str = "simple") -> dict:
    if not api_key: return {}
    payload = {
        "url": url,
        "formats": [
            "html",
            {"type":"json","schema":{"type":"object","properties":{"price":{"type":"string"}},"required":[]}}
        ],
        "proxy":"auto",
        "timeout": 45000,
        "device":"desktop"
    }
    if mode == "gentle":
        payload["actions"] = [
            {"type":"wait","milliseconds":800},
            {"type":"scroll","y":1200},
            {"type":"wait","milliseconds":1000},
        ]
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v2/scrape",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"},
            json=payload, timeout=75
        )
        if r.status_code >= 400: return {}
        return r.json()
    except Exception:
        return {}

def parse_image_and_price_from_v2(scrape: dict) -> Tuple[str,str,str]:
    """Return (image_url, price, html) from Firecrawl response."""
    if not scrape: return "","", ""
    data = scrape.get("data") or {}
    meta = data.get("metadata") or {}
    html = data.get("html") or ""
    img = meta.get("og:image") or meta.get("twitter:image") or meta.get("image") or ""
    price = ""
    j = data.get("json")
    if isinstance(j, dict):
        content = j.get("content") if isinstance(j.get("content"), dict) else j
        if isinstance(content, dict):
            price = (content.get("price") or "").strip()
    if (not price) and html:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                obj = json.loads(tag.string or "")
            except Exception:
                continue
            objs = obj if isinstance(obj, list) else [obj]
            for o in objs:
                t = o.get("@type")
                if t == "Product" or (isinstance(t,list) and "Product" in t):
                    offers = o.get("offers") or {}
                    if isinstance(offers, list): offers = offers[0] if offers else {}
                    p = offers.get("price") or (offers.get("priceSpecification") or {}).get("price")
                    if p:
                        price = p if str(p).startswith("$") else f"${p}"
                        break
            if price: break
        if not img:
            m = soup.find("meta", attrs={"property":"og:image"}) or soup.find("meta", attrs={"name":"og:image"})
            if m and m.get("content"): img = m["content"]
        if not price:
            m = PRICE_RE.search(soup.get_text(" ", strip=True))
            if m: price = m.group(0)
    return img or "", price or "", html or ""

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

def enrich_domain_firecrawl_v2(url: str, api_key: str) -> Tuple[str,str,str]:
    sc = firecrawl_scrape_v2(url, api_key, mode="simple")
    img, price, _ = parse_image_and_price_from_v2(sc)
    status = "fc_simple"
    if not img or not price:
        sc2 = firecrawl_scrape_v2(url, api_key, mode="gentle")
        i2, p2, _ = parse_image_and_price_from_v2(sc2)
        img = img or i2; price = price or p2
        if i2 or p2: status = "fc_gentle"
    return img, price, status

def enrich_urls(df: pd.DataFrame, url_col: str, api_key: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if url_col not in out.columns:
        if len(out.columns) >= 2: url_col = out.columns[1]
        else:
            st.error(f"URL column '{url_col}' not found."); return out

    urls = out[url_col].astype(str).fillna("").tolist()
    imgs   = out.get("scraped_image_url", pd.Series([""]*len(out))).astype(str).tolist()
    prices = out.get("price",               pd.Series([""]*len(out))).astype(str).tolist()
    status = out.get("scrape_status",       pd.Series([""]*len(out))).astype(str).tolist()

    prog = st.progress(0); box = st.empty()
    t0 = time.perf_counter(); times = []
    key = (api_key or "").strip()

    for i, u in enumerate(urls, start=1):
        u = u.strip()
        img = price = ""; st_code = ""
        with Timer() as tt:
            if key and u:
                img, price, st_code = enrich_domain_firecrawl_v2(u, key)
            if (not img or not price) and u:
                r = requests_get(u)
                if r and r.text:
                    i2, p2 = pick_image_and_price_bs4(r.text, u)
                    img = img or i2; price = price or p2
                    st_code = (st_code + "+bs4_ok") if st_code else "bs4_ok"
                else:
                    st_code = (st_code + "+fetch_failed") if st_code else "fetch_failed"
        imgs[i-1] = img; prices[i-1] = price; status[i-1] = st_code
        times.append(tt.dt)
        avg = sum(times)/len(times)
        eta = int((len(urls)-i)*avg)
        box.write(f"Processed {i}/{len(urls)} • last {tt.dt:.2f}s • avg {avg:.2f}s/link • ETA ~{eta}s")
        prog.progress(i/len(urls))
        time.sleep(0.1)

    out["scraped_image_url"] = imgs
    out["price"] = prices
    out["scrape_status"] = status
    return out

# ========================= UI =========================
with st.sidebar:
    st.subheader("Firecrawl (optional)")
    api_key_input = st.text_input("FIRECRAWL_API_KEY", value=os.getenv("FIRECRAWL_API_KEY",""), type="password")
    st.caption("Leave blank to use the built-in parser only (no credits used).")

tab1, tab2, tab3 = st.tabs([
    "1) Extract from PDF (pages + Tags + Position + full titles)",
    "2) Enrich CSV (Image URL + Price)",
    "3) Test single URL",
])

# --- Tab 1 ---
with tab1:
    st.caption("Build a page→Tags table, then extract ALL links from those pages. Each row starts at its own numbered item, with Position captured.")
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
    default_df = pd.DataFrame([{"page": "", "Tags": ""}])
    mapping_df = st.data_editor(
        default_df, num_rows="dynamic", use_container_width=True, key="page_tag_editor",
        column_config={
            "page": st.column_config.TextColumn("page", help="Page number (1-based)"),
            "Tags": st.column_config.TextColumn("Tags", help="Room name for that page"),
        }
    )

    only_listed = st.checkbox("Only extract pages listed above", value=True)
    run1 = st.button("Extract", type="primary", disabled=(pdf_file is None), key="extract_btn_v3")

    if run1 and pdf_file:
        page_to_tag = {}
        if num_pages is None:
            try:
                num_pages = len(fitz.open("pdf", pdf_file.getvalue()))
            except:
                num_pages = None
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
        with st.spinner("Extracting links, positions & titles…"):
            df = extract_links_by_pages(
                pdf_bytes, page_to_tag, only_listed_pages=only_listed
            )
        if df.empty:
            st.info("No links found. Verify the PDF links are live (not just images).")
        else:
            st.success(f"Extracted {len(df)} row(s).")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="canva_links_with_position.csv",
                mime="text/csv"
            )

# --- Tab 2 ---
with tab2:
    st.caption("Upload a CSV containing product URLs (column name 'Product URL' or pick another).")
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
                    df_out = enrich_urls(df_in, url_col, api_key_input)
                st.success("Enriched! ✅")
                st.dataframe(df_out, use_container_width=True)
                st.caption(df_out["scrape_status"].value_counts(dropna=False).to_frame("count"))
                out_csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download enriched CSV",
                    data=out_csv,
                    file_name="links_enriched.csv",
                    mime="text/csv",
                )
    else:
        st.info("Upload a CSV to enrich. Tip: you can feed the CSV from Tab 1 after you add product URLs.")

# --- Tab 3 ---
with tab3:
    st.caption("Test a single product URL (Firecrawl v2 first, then local parser fallback).")
    test_url = st.text_input("Product URL to test", "https://www.wayfair.com/")
    if st.button("Run test", key="single_test_btn"):
        img = price = ""; status = ""
        if api_key_input:
            i1, p1, st1 = enrich_domain_firecrawl_v2(test_url, api_key_input)
            img = img or i1; price = price or p1; status = st1 or status
        if not img or not price:
            r = requests_get(test_url)
            if r and r.text:
                i2, p2 = pick_image_and_price_bs4(r.text, test_url)
                img = img or i2; price = price or p2
                status = (status + "+bs4_ok") if status else "bs4_ok"
            else:
                status = (status + "+fetch_failed") if status else "fetch_failed"

        st.write("**Status:**", status or "unknown")
        st.write("**Image URL:**", img or "—")
        st.write("**Price:**", price or "—")
        if img:
            st.image(img, caption="Preview", use_container_width=True)


