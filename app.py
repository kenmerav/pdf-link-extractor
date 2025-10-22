# app.py
import os, io, json, re, requests, time
from typing import Optional, Tuple
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

# ===================== Streamlit setup =====================
st.set_page_config(page_title="Spec Link Extractor & Enricher", layout="wide")
st.title("🧰 Spec Link Extractor & Enricher")

# ----------------- Progress timer helper -----------------
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0

# ===================== PDF → Links extractor =====================
def extract_links_from_pdf(pdf_bytes: bytes) -> pd.DataFrame:
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

# ===================== Common helpers =====================
PRICE_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")  # $1,234.56
UA = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

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

def canonicalize_url(u: str) -> str:
    """Strip tracking params so sites return cleaner metadata."""
    try:
        p = urlparse(u)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith(("utm_", "gclid", "gbraid", "wbraid", "mc_", "msclkid"))]
        return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), ""))  # drop fragment
    except Exception:
        return u

def best_from_srcset(srcset_value: str) -> str:
    """Pick the largest candidate URL from a srcset string."""
    best_url, best_w = "", -1
    for part in (srcset_value or "").split(","):
        s = part.strip()
        if not s:
            continue
        pieces = s.split()
        url = pieces[0]
        w = -1
        if len(pieces) > 1 and pieces[1].endswith("w"):
            try:
                w = int(pieces[1][:-1])
            except Exception:
                w = -1
        if w > best_w:
            best_w, best_url = w, url
    return best_url

def best_from_picture(picture_tag) -> str:
    """Return the largest URL from <picture><source srcset=...>."""
    if not picture_tag:
        return ""
    best_url, best_w = "", -1
    for src in picture_tag.find_all("source"):
        ss = src.get("srcset") or src.get("data-srcset") or ""
        for part in ss.split(","):
            s = part.strip()
            if not s:
                continue
            pieces = s.split()
            url = pieces[0]
            w = -1
            if len(pieces) > 1 and pieces[1].endswith("w"):
                try:
                    w = int(pieces[1][:-1])
                except Exception:
                    w = -1
            if w > best_w:
                best_w, best_url = w, url
    return best_url

# ---------- Scene7 (Lumens) upgrader ----------
def _update_query(url: str, add: dict, remove_keys: tuple = ()) -> str:
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    for k in remove_keys:
        q.pop(k, None)
    q.update({k: str(v) for k, v in add.items()})
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), ""))

def _head_size(url: str, timeout: int = 10) -> tuple[int, str]:
    try:
        r = requests.head(url, headers=UA, allow_redirects=True, timeout=timeout)
        if r.status_code >= 400:
            r = requests.get(url, headers=UA, stream=True, allow_redirects=True, timeout=timeout)
            if r.status_code >= 400:
                return -1, url
        length = int(r.headers.get("Content-Length") or -1)
        return length, r.url
    except Exception:
        return -1, url

def lumens_upgrade_scene7_url(url: str) -> str:
    """Probe zoom/preset/wide variants and pick the largest working Scene7 image."""
    if not isinstance(url, str) or "is/image" not in url:
        return url
    host = urlparse(url).netloc.lower()
    if "lumens" not in host and "scene7" not in host and "is/image" not in url:
        return url

    candidates = [url]
    if "$" in url:
        candidates.extend([
            url.replace("$Lumens.com-PDP-large$", "$Lumens.com-PDP-zoom$"),
            url.replace("$Lumens.com-PDP-large$", "$Lumens.com-Product-Zoom$"),
            url.replace("$Lumens.com-PDP-large$", "$Lumens.com-zoom$"),
        ])

    base_no_preset = url
    try:
        if "?" in url:
            path, q = url.split("?", 1)
            if q.startswith("$") and q.endswith("$"):
                base_no_preset = path
    except Exception:
        pass

    for qs in [{"wid": 2400, "qlt": 90, "fmt": "jpg"},
               {"wid": 2000, "qlt": 90, "fmt": "jpg"},
               {"wid": 1600, "qlt": 90, "fmt": "jpg"}]:
        candidates.append(_update_query(base_no_preset, qs))
    candidates.append(_update_query(url, {"wid": 2000, "qlt": 90}))

    seen, uniq = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    best_len, best_url = -1, url
    for cand in uniq:
        clen, final = _head_size(cand)
        if clen > best_len:
            best_len, best_url = clen, final
    return best_url

# ===================== Firecrawl v2 (REST) helpers =====================
def firecrawl_scrape_v2(url: str, api_key: str, mode: str = "simple") -> dict:
    """Firecrawl /v2/scrape (simple or gentle)."""
    if not api_key:
        return {}
    payload = {
        "url": url,
        "formats": [
            "html",
            {"type": "json", "schema": {
                "type": "object",
                "properties": {"price": {"type": "string"}},
                "required": []
            }}
        ],
        "proxy": "auto",
        "timeout": 45000,
    }
    if mode == "gentle":
        payload["actions"] = [
            {"type": "wait", "milliseconds": 800},
            {"type": "scroll", "y": 1200},
            {"type": "wait", "milliseconds": 1200},
        ]
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v2/scrape",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=75
        )
        if r.status_code >= 400:
            return {}
        return r.json()
    except Exception:
        return {}

def firecrawl_scrape_v2_aggressive(url: str, api_key: str) -> dict:
    """Aggressive: wait + scroll + evaluate JS to extract srcset/lazy images + price."""
    if not api_key:
        return {}
    payload = {
        "url": url,
        "formats": [
            "html",
            {"type": "json", "schema": {
                "type": "object",
                "properties": {"price": {"type": "string"}},
                "required": []
            }}
        ],
        "proxy": "auto",
        "timeout": 60000,
        "device": "desktop",
        "actions": [
            {"type": "wait", "milliseconds": 800},
            {"type": "scroll", "y": 800},
            {"type": "wait", "milliseconds": 1000},
            {"type": "waitForSelector", "selector": "picture source[srcset], img[srcset], img[data-src], img[data-srcset]", "timeout": 8000},
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

                  document.querySelectorAll('meta[property="og:image"],meta[name="og:image"],meta[property="og:image:url"],meta[property="og:image:secure_url"],meta[name="twitter:image"]').forEach(m => {
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
                          if (!out.price && obj.offers) {
                            const offers = Array.isArray(obj.offers) ? obj.offers[0] : obj.offers;
                            const p = offers && (offers.price || (offers.priceSpecification && offers.priceSpecification.price));
                            if (p) out.price = String(p);
                          }
                        }
                      });
                    } catch {}
                  });

                  document.querySelectorAll("picture source[srcset], source[srcset]").forEach(src => {
                    const best = bestFromSrcset(src.getAttribute("srcset"));
                    if (best) out.images.push(best);
                  });

                  document.querySelectorAll("img").forEach(img => {
                    let cand = img.getAttribute("src") || img.getAttribute("data-src") || "";
                    if (!cand) {
                      const ss = img.getAttribute("srcset") || img.getAttribute("data-srcset");
                      if (ss) cand = bestFromSrcset(ss);
                    }
                    cand = cand || img.getAttribute("data-zoom-image") || img.getAttribute("data-large_image") || "";
                    if (cand) out.images.push(cand);
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

def parse_image_and_price_from_v2(scrape: dict) -> Tuple[str, str]:
    """Pull og:image + price from a /v2/scrape response."""
    if not scrape: return "", ""
    data = scrape.get("data") or {}
    meta = data.get("metadata") or {}
    img = meta.get("og:image") or meta.get("twitter:image") or meta.get("image") or ""

    price = ""
    j = data.get("json")
    if isinstance(j, dict):
        content = j.get("content") if isinstance(j.get("content"), dict) else j
        if isinstance(content, dict):
            price = (content.get("price") or "").strip()

    if not price or not img:
        html = data.get("html") or ""
        soup = BeautifulSoup(html or "", "lxml")
        if not price:
            for tag in soup.find_all("script", type="application/ld+json"):
                try:
                    obj = json.loads(tag.string or "")
                    objs = obj if isinstance(obj, list) else [obj]
                    for o in objs:
                        t = o.get("@type")
                        if t == "Product" or (isinstance(t, list) and "Product" in t):
                            offers = o.get("offers") or {}
                            if isinstance(offers, list): offers = offers[0] if offers else {}
                            p = offers.get("price") or (offers.get("priceSpecification") or {}).get("price")
                            if p:
                                price = p if str(p).startswith("$") else f"${p}"
                except Exception:
                    pass
        if not img:
            m = soup.find("meta", attrs={"property":"og:image"}) or soup.find("meta", attrs={"name":"og:image"})
            if m and m.get("content"):
                img = m["content"]

    if not price and (data.get("html") or ""):
        m = PRICE_RE.search(BeautifulSoup(data.get("html"), "lxml").get_text(" ", strip=True))
        if m: price = m.group(0)

    return img or "", price or ""

def parse_image_and_price_from_v2_aggressive(scrape: dict) -> Tuple[str, str]:
    """Read images/price returned by the evaluate action."""
    if not scrape:
        return "", ""
    data = scrape.get("data") or {}
    outputs = data.get("actionsOutput") or data.get("actions") or data.get("evaluate") or []
    if isinstance(outputs, dict):
        outputs = [outputs]
    images, price = [], ""

    def ingest(payload):
        nonlocal images, price
        if not payload: return
        obj = payload.get("output") if isinstance(payload, dict) else payload
        if isinstance(obj, dict):
            if "images" in obj and isinstance(obj["images"], list):
                images.extend([x for x in obj["images"] if isinstance(x, str) and x])
            if not price and isinstance(obj.get("price"), str):
                price = obj["price"].strip()

    for p in outputs:
        ingest(p)

    j = data.get("json")
    if isinstance(j, dict):
        content = j.get("content") if isinstance(j.get("content"), dict) else j
        if isinstance(content, dict) and not price and isinstance(content.get("price"), str):
            price = content["price"].strip()

    img = next((u.strip() for u in images if isinstance(u, str) and u.strip()), "")
    return img, price

# ===================== Domain enrichers =====================
def enrich_domain_firecrawl_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    sc = firecrawl_scrape_v2(url, api_key, mode="simple")
    img, price = parse_image_and_price_from_v2(sc)
    status = "firecrawl_v2_simple"
    if not img or not price:
        sc2 = firecrawl_scrape_v2(url, api_key, mode="gentle")
        i2, p2 = parse_image_and_price_from_v2(sc2)
        img = img or i2
        price = price or p2
        if i2 or p2:
            status = "firecrawl_v2_gentle"
    return img, price, status

def enrich_wayfair_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    return enrich_domain_firecrawl_v2(url, api_key)

def enrich_ferguson_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    return enrich_domain_firecrawl_v2(url, api_key)

# ---------- Lumens-specific parsing (handles <picture><source>, lazy, etc.) ----------
def parse_image_and_price_lumens(scrape: dict) -> Tuple[str, str]:
    if not scrape:
        return "", ""
    data = scrape.get("data") or {}
    meta = data.get("metadata") or {}
    html = data.get("html") or ""
    soup = BeautifulSoup(html or "", "lxml")

    # IMAGE: meta fallbacks
    img = (meta.get("og:image") or meta.get("og:image:secure_url") or meta.get("og:image:url")
           or meta.get("twitter:image") or meta.get("image") or "")

    # JSON-LD Product.image
    if not img:
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                obj = json.loads(tag.string or "")
                objs = obj if isinstance(obj, list) else [obj]
                for o in objs:
                    t = o.get("@type")
                    if t == "Product" or (isinstance(t, list) and "Product" in t):
                        im = o.get("image")
                        if isinstance(im, list) and im: img = im[0]
                        elif isinstance(im, str) and im: img = im
                        if img: break
                if img: break
            except Exception:
                pass

    # <picture><source srcset="...">
    if not img:
        scope = soup.select_one('[class*="pdp"], [id*="pdp"], [class*="gallery"], [id*="gallery"]') or soup
        best_pic, best_score = None, -1
        for pict in scope.find_all("picture"):
            s = " ".join([str(x) for x in (pict.get("class") or []) + [pict.get("id") or ""]]).lower()
            score = 5 if any(k in s for k in ("product", "hero", "main", "primary", "gallery", "pdp")) else 0
            cand_url = best_from_picture(pict)
            if cand_url: score += 3
            if score > best_score:
                best_score, best_pic = score, pict
        if best_pic:
            cand = best_from_picture(best_pic)
            if cand: img = cand

    # <img> src/srcset/data-*
    if not img:
        scope = soup.select_one('[class*="pdp"], [id*="pdp"], [class*="gallery"], [id*="gallery"]') or soup
        candidates = []
        for imgtag in scope.find_all("img"):
            cand = imgtag.get("src") or imgtag.get("data-src") or ""
            if not cand:
                ss = imgtag.get("srcset") or imgtag.get("data-srcset")
                if ss: cand = best_from_srcset(ss)
            cand = cand or imgtag.get("data-zoom-image") or imgtag.get("data-large_image")
            if cand:
                score = 0
                s = " ".join([str(x) for x in (imgtag.get("class") or []) + [imgtag.get("id") or ""]]).lower()
                if any(k in s for k in ("product", "hero", "primary", "main", "zoom")): score += 5
                if any(k in (cand or "").lower() for k in ("cloudinary", "scene7", "akamai", "cdn", "lumens", "images", "media")): score += 3
                try:
                    w = int(imgtag.get("width") or 0)
                    if w >= 600: score += 2
                except Exception:
                    pass
                candidates.append((score, cand))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            img = candidates[0][1]

    # <link rel="preload" as="image">
    if not img:
        preload = soup.find("link", rel=lambda v: v and "preload" in v, attrs={"as": "image"})
        if preload and preload.get("href"):
            img = preload["href"]

    # PRICE
    price = ""
    j = data.get("json")
    if isinstance(j, dict):
        content = j.get("content") if isinstance(j.get("content"), dict) else j
        if isinstance(content, dict):
            price = (content.get("price") or "").strip()

    if not price:
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
            if price: break

    if not price:
        m = soup.find("meta", attrs={"itemprop": "price"}) or soup.find("meta", attrs={"property": "product:price:amount"})
        if m and m.get("content"):
            val = m["content"]
            price = val if str(val).startswith("$") else f"${val}"

    if not price:
        t = soup.get_text(" ", strip=True)
        m = PRICE_RE.search(t)
        if m:
            price = m.group(0)

    return (img or ""), (price or "")

def enrich_lumens_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    u = canonicalize_url(url)

    # simple
    sc = firecrawl_scrape_v2(u, api_key, mode="simple")
    img, price = parse_image_and_price_lumens(sc)
    status = "firecrawl_v2_simple"

    # gentle
    if not img or not price:
        sc2 = firecrawl_scrape_v2(u, api_key, mode="gentle")
        i2, p2 = parse_image_and_price_lumens(sc2)
        if i2 or p2:
            img = img or i2
            price = price or p2
            status = "firecrawl_v2_gentle"

    # aggressive (evaluate)
    if not img:
        sc3 = firecrawl_scrape_v2_aggressive(u, api_key)
        i3, p3 = parse_image_and_price_from_v2_aggressive(sc3)
        if i3 or p3:
            img = img or i3
            price = price or p3
            status = (status + "+v2_aggressive") if status else "v2_aggressive"

    return img, price, status or "unknown"

# ===================== BeautifulSoup fallback (with Lumens hook) =====================
def pick_image_and_price_bs4(html: str, base_url: str) -> Tuple[str, str]:
    """og/twitter → JSON-LD → hero <img> → meta → visible $; then Scene7 upgrade if applicable."""
    soup = BeautifulSoup(html or "", "lxml")

    # Image: og/twitter
    img_url = ""
    for sel in [("meta", {"property":"og:image"}), ("meta", {"name":"og:image"}),
                ("meta", {"name":"twitter:image"}), ("meta", {"property":"twitter:image"})]:
        tag = soup.find(*sel)
        if tag and tag.get("content"):
            img_url = urljoin(base_url, tag["content"]); break

    # Image: JSON-LD Product.image
    if not img_url:
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(tag.string or "")
                objs = data if isinstance(data, list) else [data]
                for obj in objs:
                    t = obj.get("@type")
                    if t == "Product" or (isinstance(t, list) and "Product" in t):
                        im = obj.get("image")
                        if isinstance(im, list) and im:
                            img_url = urljoin(base_url, im[0]); break
                        if isinstance(im, str) and im:
                            img_url = urljoin(base_url, im); break
                if img_url: break
            except Exception:
                pass

    # Prefer direct hero <img> with src/data-src (handles Lumens lazies)
    if not img_url:
        hero = soup.select_one('img[src*="/is/image/"], img[data-src*="/is/image/"]')
        if hero:
            img_url = urljoin(base_url, hero.get("src") or hero.get("data-src") or "")

    # Last-resort image: first <img>
    if not img_url:
        anyimg = soup.find("img", src=True)
        if anyimg: img_url = urljoin(base_url, anyimg["src"])

    # Price: JSON-LD offers.price
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

    # Price: meta itemprop/property
    if not price:
        meta_price = soup.find("meta", attrs={"itemprop":"price"}) or \
                     soup.find("meta", attrs={"property":"product:price:amount"})
        if meta_price and meta_price.get("content"):
            val = meta_price["content"]
            price = f"${val}" if not str(val).startswith("$") else str(val)

    # Price: visible pattern
    if not price:
        m = PRICE_RE.search(soup.get_text(" ", strip=True))
        if m: price = m.group(0)

    # Upgrade Scene7/Lumens URLs to largest working variant
    if img_url and ("images.lumens.com/is/image" in img_url or "/is/image/" in img_url):
        img_url = lumens_upgrade_scene7_url(img_url)

    return img_url or "", price or ""

# ===================== Sidebar (API key) =====================
with st.sidebar:
    st.subheader("Firecrawl (optional)")
    api_key_input = st.text_input(
        "FIRECRAWL_API_KEY",
        value=os.getenv("FIRECRAWL_API_KEY", ""),
        type="password",
        help="Leave blank to use the built-in parser only (no credits used)."
    )
    st.caption("Tip: In Streamlit Cloud, put this in **Settings → Secrets**.")

# ===================== Tabs =====================
tab1, tab2, tab3 = st.tabs([
    "1) Extract from PDF",
    "2) Enrich CSV (Image URL + Price)",
    "3) Test single URL"
])

# ===================== Batch enrich =====================
def enrich_urls(df: pd.DataFrame, url_col: str, api_key: Optional[str]) -> pd.DataFrame:
    """
    Add scraped_image_url and price; Firecrawl v2 first (if key), fallback to bs4,
    with live progress bar + ETA.
    """
    out = df.copy()
    if url_col not in out.columns:
        if len(out.columns) >= 2:
            url_col = out.columns[1]
        else:
            st.error(f"URL column '{url_col}' not found.")
            return out

    urls = out[url_col].astype(str).fillna("").tolist()

    if "scraped_image_url" in out.columns and "price" in out.columns:
        mask_done = out["scraped_image_url"].astype(str).ne("") & out["price"].astype(str).ne("")
    else:
        mask_done = pd.Series([False]*len(out))

    idxs = [i for i, done in enumerate(mask_done) if not done]

    imgs   = out.get("scraped_image_url", pd.Series([""]*len(out))).astype(str).tolist()
    prices = out.get("price",               pd.Series([""]*len(out))).astype(str).tolist()
    status = out.get("scrape_status",       pd.Series([""]*len(out))).astype(str).tolist()

    api_key = (api_key or "").strip()

    prog = st.progress(0)
    status_box = st.empty()
    t_start = time.perf_counter()
    per_link_times = []

    for k, i in enumerate(idxs, start=1):
        u = urls[i].strip()
        if not u:
            imgs[i] = ""; prices[i] = ""; status[i] = "no_url"
            prog.progress(k/len(idxs))
            continue

        with Timer() as t:
            img = price = ""; st_code = ""

            # Firecrawl v2 (domain-aware)
            if api_key:
                if "lumens.com" in u:
                    img, price, st_code = enrich_lumens_v2(u, api_key)
                elif "fergusonhome.com" in u:
                    img, price, st_code = enrich_ferguson_v2(u, api_key)
                elif "wayfair.com" in u:
                    img, price, st_code = enrich_wayfair_v2(u, api_key)
                else:
                    img, price, st_code = enrich_domain_firecrawl_v2(u, api_key)

            # Fallback: requests + BeautifulSoup
            if not img or not price:
                r = requests_get(u)
                if r and r.text:
                    img2, price2 = pick_image_and_price_bs4(r.text, u)
                    img = img or img2
                    price = price or price2
                    st_code = (st_code + "+bs4_ok") if st_code else "bs4_ok"
                else:
                    st_code = (st_code + "+fetch_failed") if st_code else "fetch_failed"

            # Upgrade Scene7 if applicable
            if img and ("images.lumens.com/is/image" in img or "/is/image/" in img):
                img = lumens_upgrade_scene7_url(img)

            imgs[i] = img
            prices[i] = price
            status[i] = st_code

        per_link_times.append(t.dt)
        avg = sum(per_link_times) / max(len(per_link_times), 1)
        remaining = (len(idxs) - k) * avg
        status_box.write(
            f"Processed {k}/{len(idxs)} • last {t.dt:.2f}s • avg {avg:.2f}s/link • ETA ~{int(remaining)}s"
        )
        prog.progress(k/len(idxs))
        time.sleep(0.10)

    total = time.perf_counter() - t_start
    if idxs:
        status_box.write(f"Done {len(idxs)} link(s) in {total:.1f}s • avg {(total/len(idxs)):.2f}s/link")
    else:
        status_box.write("Nothing to do — all rows already enriched.")

    out["scraped_image_url"] = imgs
    out["price"] = prices
    out["scrape_status"] = status
    return out

# ===================== Tab 1: PDF → Links =====================
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

# ===================== Tab 2: Enrich CSV =====================
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

# ===================== Tab 3: Test a single URL =====================
with tab3:
    st.caption("Paste a single product URL and test the enrichment (Firecrawl v2 first, then fallback).")
    test_url = st.text_input("Product URL to test", "https://www.fergusonhome.com/product/summary/1871316?uid=4421090")

    if st.button("Run test", key="single_test_btn"):
        test_url = canonicalize_url(test_url)
        img = price = ""; status = ""
        if api_key_input:
            if "lumens.com" in test_url:
                img, price, status = enrich_lumens_v2(test_url, api_key_input)
            elif "fergusonhome.com" in test_url:
                img, price, status = enrich_ferguson_v2(test_url, api_key_input)
            elif "wayfair.com" in test_url:
                img, price, status = enrich_wayfair_v2(test_url, api_key_input)
            else:
                img, price, status = enrich_domain_firecrawl_v2(test_url, api_key_input)

        if not img or not price:
            r = requests_get(test_url)
            if r and r.text:
                i2, p2 = pick_image_and_price_bs4(r.text, test_url)
                img = img or i2
                price = price or p2
                status = (status + "+bs4_ok") if status else "bs4_ok"
            else:
                status = (status + "+fetch_failed") if status else "fetch_failed"

        # Upgrade Scene7 if applicable
        if img and ("images.lumens.com/is/image" in img or "/is/image/" in img):
            img = lumens_upgrade_scene7_url(img)

        st.write("**Status:**", status or "unknown")
        st.write("**Image URL:**", img or "—")
        st.write("**Price:**", price or "—")
        if img:
            st.image(img, caption="Preview", use_container_width=True)
