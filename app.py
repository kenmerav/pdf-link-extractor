# app.py — Full app:
# 1) Canva PDF → Page→Tags + strict per-link title capture + Position + Type/QTY/Finish/Size
# 2) Enrich CSV (Firecrawl v2 → bs4 fallback) to get image URL + price
# 3) Test a single URL
# Includes hardened Lumens handler.

import os, re, json, time, requests
from typing import Optional, Tuple
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse, urljoin, parse_qsl, urlencode

# ========================= App setup =========================
st.set_page_config(page_title="Spec Link Toolkit", layout="wide")
st.title("🧰 Spec Link Toolkit")

PRICE_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
}

def requests_get(url: str, timeout: int = 20, retries: int = 2) -> Optional[requests.Response]:
    for _ in range(retries + 1):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if 200 <= r.status_code < 300:
                return r
        except Exception:
            pass
        time.sleep(0.25)
    return None

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0


# ========================= TAB 1: Canva PDF extractor =========================
# Parsing rules for Type / QTY / Finish / Size
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

    if not fields["Type"]:
        for tok in parts:
            if ":" in tok:
                continue
            if tok.strip():
                fields["Type"] = tok.strip()
                break

    if fields["Size"]:
        fields["Size"] = re.sub(r"\s*[xX]\s*", " x ", fields["Size"]).strip()

    return fields

# STRICT per-link text capture: only words whose CENTER lies inside link box
def extract_link_title_strict(page, rect, pad_px: float = 2.0, band_px: float = 18.0) -> str:
    """
    Capture text for this link only.
    - Primary: words whose center lies inside (rect ± pad_px)
    - If empty: look in a thin horizontal band around the rect (for cases where
      the link is only on an image or underline but the label sits just outside).
    - Words are taken from raw block->line->span order for stable layout order.
    """
    import fitz as _fitz
    def norm(s: str) -> str:
        s = re.sub(r"\s*\|\s*", " | ", s)
        s = re.sub(r"\s*;\s*", "; ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()

    if not rect:
        return ""

    r = _fitz.Rect(rect).normalize()
    R = _fitz.Rect(r.x0 - pad_px, r.y0 - pad_px, r.x1 + pad_px, r.y1 + pad_px)

    raw = page.get_text("rawdict") or {}
    words = []
    for blk in raw.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                tx = span.get("text", "")
                bbox = span.get("bbox", None)
                if not tx or not bbox:
                    continue
                x0, y0, x1, y1 = bbox
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                words.append((cx, cy, x0, y0, x1, y1, tx))

    kept = [(y0, x0, tx) for cx, cy, x0, y0, x1, y1, tx in words
            if R.contains(_fitz.Point(cx, cy)) and tx]

    if not kept:
        band = _fitz.Rect(r.x0, r.y0 - band_px, r.x1, r.y1 + band_px)
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

# Position is only at the START of this link’s text (we do not read past the link)
POS_AT_START = re.compile(r'^\s*(\d{1,3})[.\u2024\u00B7]?\s*')
# next bullet *inside* a captured string (to trim accidental merges)
NEXT_BULLET = re.compile(r'\s(\d{1,3})[.\u2024\u00B7](?=\s|[A-Z])')

def split_position_and_title_start(raw: str) -> tuple[str, str]:
    """
    Pull Position at the start of this link's text; if another bullet appears
    later in the same string, cut at that bullet to avoid merging.
    """
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

def extract_links_by_pages(
    pdf_bytes: bytes,
    page_to_tag: dict[int, str] | None,
    only_listed_pages: bool = True,
    pad_px: float = 2.0,
    band_px: float = 18.0,
    require_leading_number: bool = False,
) -> pd.DataFrame:
    doc = fitz.open("pdf", pdf_bytes)
    rows = []
    listed = set(page_to_tag.keys()) if page_to_tag else set()

    for pidx, page in enumerate(doc, start=1):
        if only_listed_pages and page_to_tag and pidx not in listed:
            continue
        tag_value = (page_to_tag or {}).get(pidx, "")

        for lnk in page.get_links():
            uri = (lnk.get("uri") or "").strip()
            if not uri.lower().startswith(("http://", "https://")):
                continue

            rect = lnk.get("from")
            raw = extract_link_title_strict(page, rect, pad_px=pad_px, band_px=band_px)
            position, title = split_position_and_title_start(raw)

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
                "link_url": uri,     # full URL
                "link_text": title,  # per-link, trimmed if a next bullet was detected
            })

    return pd.DataFrame(rows)


# ========================= TAB 2/3: Enricher (Firecrawl v2 + bs4) =========================
def firecrawl_scrape_v2(url: str, api_key: str, mode: str = "simple") -> dict:
    if not api_key:
        return {}
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
        if r.status_code >= 400:
            return {}
        return r.json()
    except Exception:
        return {}

def parse_image_and_price_from_v2(scrape: dict) -> Tuple[str, str, str]:
    if not scrape:
        return "", "", ""
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
                if t == "Product" or (isinstance(t, list) and "Product" in t):
                    offers = o.get("offers") or {}
                    if isinstance(offers, list): offers = offers[0] if offers else {}
                    p = offers.get("price") or (offers.get("priceSpecification") or {}).get("price")
                    if p:
                        price = p if str(p).startswith("$") else f"${p}"
                        break
            if price:
                break
        if not img:
            m = soup.find("meta", attrs={"property":"og:image"}) or soup.find("meta", attrs={"name":"og:image"})
            if m and m.get("content"):
                img = m["content"]
        if not price:
            m = PRICE_RE.search(soup.get_text(" ", strip=True))
            if m:
                price = m.group(0)
    return img or "", price or "", html or ""

def pick_image_and_price_bs4(html: str, base_url: str) -> Tuple[str, str]:
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

def enrich_domain_firecrawl_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    sc = firecrawl_scrape_v2(url, api_key, mode="simple")
    img, price, _ = parse_image_and_price_from_v2(sc)
    status = "fc_simple"
    if not img or not price:
        sc2 = firecrawl_scrape_v2(url, api_key, mode="gentle")
        i2, p2, _ = parse_image_and_price_from_v2(sc2)
        img = img or i2; price = price or p2
        if i2 or p2: status = "fc_gentle"
    return img, price, status


# ========================= LUMENS — hardened path =========================
def _canonicalize_lumens(u: str) -> str:
    """Strip tracking params; keep color/variant if present."""
    try:
        p = urlparse(u)
        keep = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            if k.lower().startswith(("dwvar_", "sku", "variant")):
                keep.append((k, v))
        return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(keep), ""))
    except Exception:
        return u

def _best_from_srcset(ss: str) -> str:
    best_url, best_w = "", -1
    for part in (ss or "").split(","):
        p = part.strip().split()
        if not p: 
            continue
        url = p[0]
        w = -1
        if len(p) > 1 and p[1].endswith("w"):
            try:
                w = int(p[1][:-1])
            except Exception:
                pass
        if w > best_w:
            best_w, best_url = w, url
    return best_url

def _lumens_image_from_html(html: str) -> str:
    """Prefer images.lumens.com is/image URL; fall back to hero <picture>/<img>."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")

    # 1) <meta> og/twitter
    for sel in [
        ("meta", {"property": "og:image"}),
        ("meta", {"name": "og:image"}),
        ("meta", {"name": "twitter:image"}),
        ("meta", {"property": "twitter:image"}),
    ]:
        tag = soup.find(*sel)
        if tag and tag.get("content"):
            v = tag["content"].strip()
            if v:
                return v

    # 2) JSON-LD Product.image
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            items = data if isinstance(data, list) else [data]
            for obj in items:
                t = obj.get("@type")
                if t == "Product" or (isinstance(t, list) and "Product" in t):
                    img = obj.get("image")
                    if isinstance(img, list) and img:
                        return img[0]
                    if isinstance(img, str) and img:
                        return img
        except Exception:
            pass

    # 3) Prefer Lumens dynamic server “is/image”
    for imgtag in soup.find_all("img"):
        cand = (
            imgtag.get("data-enlarged") or
            imgtag.get("data-zoom-image") or
            imgtag.get("data-large_image") or
            imgtag.get("data-src") or
            imgtag.get("src") or
            ""
        )
        if "images.lumens.com/is/image" in (cand or ""):
            return cand

        ss = imgtag.get("srcset") or imgtag.get("data-srcset")
        if ss:
            best = _best_from_srcset(ss)
            if "images.lumens.com/is/image" in best:
                return best

    # 4) <picture><source srcset>
    best_url = ""
    best_score = -1
    for pic in soup.find_all("picture"):
        cand = ""
        for src in pic.find_all("source"):
            b = _best_from_srcset(src.get("srcset") or "")
            if b:
                cand = b
        if not cand:
            continue
        score = 0
        cstr = " ".join((pic.get("id") or "", " ".join(pic.get("class") or []))).lower()
        if any(k in cstr for k in ("pdp", "product", "hero", "main", "gallery")):
            score += 5
        if "images.lumens.com/is/image" in cand:
            score += 3
        if score > best_score:
            best_score, best_url = score, cand
    if best_url:
        return best_url

    # 5) Fallback first <img>
    anyimg = soup.find("img", src=True)
    return anyimg["src"].strip() if anyimg else ""

def firecrawl_scrape_v2_aggressive(url: str, api_key: str) -> dict:
    """Aggressive Firecrawl with waits + JS evaluate to pull lazy/srcset/enlarged."""
    if not api_key:
        return {}
    payload = {
        "url": url,
        "formats": [
            "html",
            {"type": "json", "schema": {"type": "object", "properties": {"price": {"type": "string"}}, "required": []}}
        ],
        "render": True,
        "device": "desktop",
        "proxy": "auto",
        "timeout": 60000,
        "actions": [
            {"type": "wait", "milliseconds": 900},
            {"type": "scroll", "y": 1200},
            {"type": "wait", "milliseconds": 900},
            {"type": "waitForSelector", "selector": "picture source[srcset], img[data-src], img[srcset], img[data-enlarged], img[src]", "timeout": 8000},
            {
                "type": "evaluate",
                "script": """
                (() => {
                  function bestFromSrcset(ss) {
                    if (!ss) return "";
                    let best = ["", -1];
                    ss.split(",").forEach(part => {
                      const p = part.trim().split(/\\s+/);
                      const url = p[0] || "";
                      let w = -1;
                      if (p[1] && /w$/.test(p[1])) {
                        const n = parseInt(p[1].slice(0, -1), 10);
                        if (!isNaN(n)) w = n;
                      }
                      if (w > best[1]) best = [url, w];
                    });
                    return best[0];
                  }
                  const out = { images: [], price: "" };
                  document.querySelectorAll('meta[property="og:image"],meta[name="og:image"],meta[name="twitter:image"]').forEach(m => {
                    const v = m.getAttribute("content");
                    if (v) out.images.push(v);
                  });
                  document.querySelectorAll('script[type="application/ld+json"]').forEach(s => {
                    try {
                      const data = JSON.parse(s.textContent || "null");
                      const arr = Array.isArray(data) ? data : [data];
                      arr.forEach(obj => {
                        const t = obj && obj["@type"];
                        const isProduct = t === "Product" || (Array.isArray(t) && t.includes("Product"));
                        if (isProduct && obj.image) {
                          if (Array.isArray(obj.image)) out.images.push(...obj.image);
                          else out.images.push(obj.image);
                          const offers = Array.isArray(obj.offers) ? obj.offers[0] : obj.offers;
                          const p = offers && (offers.price || (offers.priceSpecification && offers.priceSpecification.price));
                          if (p && !out.price) out.price = String(p);
                        }
                      });
                    } catch {}
                  });
                  document.querySelectorAll("picture source[srcset]").forEach(src => {
                    const b = bestFromSrcset(src.getAttribute("srcset"));
                    if (b) out.images.push(b);
                  });
                  document.querySelectorAll("img").forEach(img => {
                    let c = img.getAttribute("data-enlarged") || img.getAttribute("data-zoom-image") || img.getAttribute("data-large_image") ||
                            img.getAttribute("data-src") || img.getAttribute("src") || "";
                    if (!c) {
                      const ss = img.getAttribute("srcset") || img.getAttribute("data-srcset");
                      if (ss) c = bestFromSrcset(ss);
                    }
                    if (c) out.images.push(c);
                  });
                  if (!out.price) {
                    const m = (document.body.innerText || "").match(/\\$\\s?\\d{1,3}(?:,\\d{3})*(?:\\.\\d{2})?/);
                    if (m) out.price = m[0];
                  }
                  const seen = new Set();
                  out.images = out.images.filter(u => (u && !seen.has(u) && seen.add(u)));
                  return out;
                })();
                """
            }
        ]
    }
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v2/scrape",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=90
        )
        if r.status_code >= 400:
            return {}
        return r.json()
    except Exception:
        return {}

def enrich_lumens_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    """
    Lumens: canonicalize → Firecrawl simple → parse html → aggressive → local bs4 fallback.
    Returns (image_url, price, status)
    """
    u = _canonicalize_lumens(url)

    sc1 = firecrawl_scrape_v2(u, api_key, mode="simple")
    img1, price1, html1 = parse_image_and_price_from_v2(sc1)
    if not img1:
        img1 = _lumens_image_from_html(html1)
    if img1 and price1:
        return img1, price1, "lumens_fc_simple"

    sc2 = firecrawl_scrape_v2_aggressive(u, api_key)
    img2, price2, html2 = parse_image_and_price_from_v2(sc2)
    if not img2:
        img2 = _lumens_image_from_html(html2)
    if img2 or price2:
        return img2, (price2 or price1), "lumens_fc_aggressive"

    r = requests_get(u)
    if r and r.text:
        img3 = _lumens_image_from_html(r.text)
        if not img3:
            i4, _ = pick_image_and_price_bs4(r.text, u)
            img3 = i4
        _, p4 = pick_image_and_price_bs4(r.text, u)
        return img3, (p4 or price1), "lumens_bs4"

    return "", "", "lumens_fail"


# ========================= Batch enrich =========================
def enrich_urls(df: pd.DataFrame, url_col: str, api_key: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if url_col not in out.columns:
        if len(out.columns) >= 2:
            url_col = out.columns[1]
        else:
            st.error(f"URL column '{url_col}' not found.")
            return out

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
                low = u.lower()
                if "lumens.com" in low:
                    img, price, st_code = enrich_lumens_v2(u, key)
                else:
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
    api_key_input = st.text_input(
        "FIRECRAWL_API_KEY",
        value=os.getenv("FIRECRAWL_API_KEY", ""),
        type="password"
    )
    st.caption("Leave blank to use the built-in parser only (no credits used).")

tab1, tab2, tab3 = st.tabs([
    "1) Extract from PDF (pages + Tags + Position + full titles)",
    "2) Enrich CSV (Image URL + Price)",
    "3) Test single URL",
])

# --- Tab 1: PDF → rows ---
with tab1:
    st.caption("Make a page→Tags table, then extract ALL web links on those pages. Each link stays on its own row.")
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
    pad_px = st.slider("Link capture pad (pixels)", 0, 10, 2, 1)
    band_px = st.slider("Nearby band height (pixels)", 0, 40, 18, 1)
    require_num = st.checkbox("Require leading number (1., 2., …) to keep a row", value=False)

    run1 = st.button("Extract", type="primary", disabled=(pdf_file is None), key="extract_btn")

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
                pdf_bytes,
                page_to_tag,
                only_listed_pages=only_listed,
                pad_px=pad_px,
                band_px=band_px,
                require_leading_number=require_num,
            )
        if df.empty:
            st.info("No links found. Verify the PDF uses live hyperlinks (not just images).")
        else:
            st.success(f"Extracted {len(df)} row(s).")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="canva_links_with_position.csv",
                mime="text/csv"
            )

# --- Tab 2: Enrich CSV ---
with tab2:
    st.caption("Upload a CSV containing product URLs (column name 'Product URL' or choose one).")
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

# --- Tab 3: Test a single URL ---
with tab3:
    st.caption("Test a single product URL (Firecrawl v2 first, then local parser fallback).")
    test_url = st.text_input("Product URL to test", "https://www.lumens.com/xavier-wall-sconce-by-crystorama-CRYP535322.html?cgid=20")
    if st.button("Run test", key="single_test_btn"):
        img = price = ""; status = ""
        if api_key_input:
            low = test_url.lower()
            if "lumens.com" in low:
                img, price, status = enrich_lumens_v2(test_url, api_key_input)
            else:
                img, price, status = enrich_domain_firecrawl_v2(test_url, api_key_input)

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
