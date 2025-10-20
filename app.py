import os, io, time, json, re, requests
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from urllib.parse import urljoin

# ---------- Your original: Extract links from PDF ----------
def extract_links_from_pdf(pdf_bytes):
    """Extract links + metadata from uploaded PDF file (bytes)."""
    doc = fitz.open("pdf", pdf_bytes)
    rows = []

    for page in doc:
        # Header area (right/top ~10% of page)
        title_full = page.get_textbox(
            fitz.Rect(page.rect.width * 0.5, 0,
                      page.rect.width, page.rect.height * 0.1)
        ).strip()
        parts = title_full.split(' ', 1)
        project = parts[0] if parts else ''
        sheet = parts[1] if len(parts) > 1 else ''

        for lnk in page.get_links():
            uri = lnk.get('uri') or ''
            if not uri.startswith(('http://', 'https://')):
                continue

            rect = fitz.Rect(lnk['from'])
            link_title = page.get_textbox(rect).strip() or ''

            rows.append({
                'Product Name': link_title,
                'Product URL': uri,
                'project': project,
                'sheet': sheet
            })

    return pd.DataFrame(rows)

# ---------- Enricher: Firecrawl (optional) + BeautifulSoup ----------
PRICE_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")  # $1,234.56
UA = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def requests_get(url, timeout=20, retries=2):
    for _ in range(retries+1):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if 200 <= r.status_code < 300:
                return r
        except Exception:
            pass
        time.sleep(0.25)
    return None

def pick_image_and_price_bs4(html: str, base_url: str):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")

    # image via og/twitter/json-ld/fallback
    img_url = ""
    for sel in [("meta", {"property":"og:image"}), ("meta", {"name":"og:image"}),
                ("meta", {"name":"twitter:image"}), ("meta", {"property":"twitter:image"})]:
        tag = soup.find(*sel)
        if tag and tag.get("content"):
            img_url = urljoin(base_url, tag["content"]); break
    if not img_url:
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(tag.string or "")
                objs = data if isinstance(data, list) else [data]
                for obj in objs:
                    t = obj.get("@type")
                    if t == "Product" or (isinstance(t, list) and "Product" in t):
                        im = obj.get("image")
                        if isinstance(im, list) and im: img_url = urljoin(base_url, im[0]); break
                        if isinstance(im, str) and im:  img_url = urljoin(base_url, im); break
                if img_url: break
            except Exception:
                pass
    if not img_url:
        anyimg = soup.find("img", src=True)
        if anyimg: img_url = urljoin(base_url, anyimg["src"])

    # price via JSON-LD, meta, or visible text
    price = ""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            objs = data if isinstance(data, list) else [data]
            for obj in objs:
                t = obj.get("@type")
                if t == "Product" or (isinstance(t, list) and "Product" in t):
                    offers = obj.get("offers") or {}
                    if isinstance(offers, list):
                        offers = offers[0] if offers else {}
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

    return img_url or "", price or ""

def firecrawl_client(api_key: str):
    if not api_key: return None
    try:
        from firecrawl import Firecrawl
        return Firecrawl(api_key=api_key)
    except Exception:
        return None

def pick_image_from_firecrawl(sc: dict) -> str:
    if not sc: return ""
    meta = sc.get("metadata") or {}
    for k in ("og:image","twitter:image","image"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for block in sc.get("data") or []:
        if block.get("type") == "json-ld":
            try:
                data = json.loads(block.get("content") or "{}")
                objs = data if isinstance(data, list) else [data]
                for obj in objs:
                    t = obj.get("@type")
                    if t == "Product" or (isinstance(t, list) and "Product" in t):
                        img = obj.get("image")
                        if isinstance(img, list) and img: return img[0]
                        if isinstance(img, str) and img.strip(): return img.strip()
            except Exception:
                pass
    imgs = sc.get("images") or []
    if imgs: return imgs[0]
    return ""

def firecrawl_extract_price(fc, url: str) -> str:
    try:
        schema = {
            "type":"object",
            "properties":{"price":{"type":"string","description":"Product price with currency symbol if shown"}},
            "required":[]
        }
        res = fc.extract({"urls":[url], "schema":schema, "render":True})
        item = (res.get("data") or [{}])[0].get("content") or {}
        price = (item.get("price") or "").strip()
        if price and not PRICE_RE.search(price):
            # normalize "USD 199" etc.
            m = PRICE_RE.search(price.replace("USD","$").replace("usd","$"))
            if m: price = m.group(0)
        return price
    except Exception:
        return ""

def enrich_urls(df: pd.DataFrame, url_col: str, api_key: str | None) -> pd.DataFrame:
    """Add scraped_image_url and price; Firecrawl first (if api_key), fallback to bs4."""
    fc = firecrawl_client(api_key)
    out = df.copy()
    if url_col not in out.columns:
        # try second column as fallback
        if len(out.columns) >= 2:
            url_col = out.columns[1]
        else:
            st.error(f"URL column '{url_col}' not found.")
            return out

    imgs, prices, status = [], [], []

    for u in out[url_col].astype(str).fillna(""):
        u = u.strip()
        if not u:
            imgs.append(""); prices.append(""); status.append("no_url"); continue

        img = price = ""; st_code = ""

        # Firecrawl first (if configured)
        if fc:
            try:
                sc = fc.scrape_url(u, params={"formats":["json","html","markdown"], "render": True, "timeout": 25000})
                img = pick_image_from_firecrawl(sc)
                price = firecrawl_extract_price(fc, u)
                st_code = "firecrawl_ok"
            except Exception:
                st_code = "firecrawl_fail"

        # Fallback to requests+bs4
        if not img or not price:
            r = requests_get(u)
            if r and r.text:
                img2, price2 = pick_image_and_price_bs4(r.text, u)
                img = img or img2
                price = price or price2
                st_code = (st_code + "+bs4_ok") if st_code else "bs4_ok"
            else:
                st_code = (st_code + "+fetch_failed") if st_code else "fetch_failed"

        imgs.append(img); prices.append(price); status.append(st_code)
        time.sleep(0.2)

    out["scraped_image_url"] = imgs
    out["price"] = prices
    out["scrape_status"] = status
    return out

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Spec Link Extractor & Enricher", layout="wide")
st.title("🧰 Spec Link Extractor & Enricher")

with st.sidebar:
    st.subheader("Firecrawl (optional)")
    api_key = st.text_input("FIRECRAWL_API_KEY", value=os.getenv("FIRECRAWL_API_KEY", ""), type="password")
    st.caption("Leave blank to use the built-in parser only (no credits used).")

tab1, tab2 = st.tabs(["1) Extract links from PDF", "2) Enrich links (Image URL + Price)"])

# --- Tab 1: Extract links from PDF ---
with tab1:
    st.caption("Upload a PDF → extract all web links with project/sheet info → download a CSV.")
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf", key="pdf_uploader")

    if uploaded_pdf:
        st.info(f"File uploaded: {uploaded_pdf.name}")
        if st.button("Extract Links", key="extract_btn"):
            with st.spinner("Extracting links..."):
                pdf_bytes = uploaded_pdf.read()
                result_df = extract_links_from_pdf(pdf_bytes)

            if result_df.empty:
                st.warning("No links found in this PDF.")
            else:
                st.success("Done! ✅")
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results CSV",
                    data=csv_bytes,
                    file_name=f"{uploaded_pdf.name}_links.csv",
                    mime="text/csv",
                )

# --- Tab 2: Enrich links (from a CSV OR from Tab 1 output you re-upload) ---
with tab2:
    st.caption("Provide a CSV with a 'Product URL' column (or the 2nd column will be used).")
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
                    df_out = enrich_urls(df_in, url_col, api_key.strip())
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
        st.info("Upload a CSV to enrich. (Tip: Use the CSV you downloaded from Tab 1.)")
