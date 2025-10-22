# app.py
import os, io, json, re, requests, time, random
from typing import Optional, Tuple
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
PRICE_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
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
    try:
        p = urlparse(u)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith(("utm_", "gclid", "gbraid", "wbraid", "msclkid", "mc_"))]
        return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), ""))
    except Exception:
        return u

def add_cache_buster(u: str) -> str:
    p = urlparse(u)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q["_fcv"] = str(int(time.time()*1000))
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), ""))

def best_from_srcset(srcset_value: str) -> str:
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

# ---------- Scene7 upgraders (Lumens + Crate/CB2) ----------
def _update_query(url: str, add: dict, remove_keys: tuple = ()) -> str:
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    for k in remove_keys:
        q.pop(k, None)
    q.update({k: str(v) for k, v in add.items()})
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), ""))

def _strip_scene7_preset(url: str) -> str:
    if "?" not in url:
        return url
    path, q = url.split("?", 1)
    q = q.strip()
    if q.startswith("$") and q.endswith("$"):
        return path
    if "$" in q:
        parts = [p for p in q.split("&") if not (p.startswith("$") and p.endswith("$"))]
        q = "&".join(parts)
        return f"{path}?{q}" if q else path
    return url

def _measure_image_dims(url: str, timeout: int = 12) -> tuple[int, int]:
    try:
        r = requests.get(url, headers=UA, stream=True, timeout=timeout)
        r.raise_for_status()
        bio = io.BytesIO()
        for chunk in r.iter_content(8192):
            if not chunk:
                break
            bio.write(chunk)
            try:
                bio.seek(0)
                with Image.open(bio) as im:
                    im.load()
                    return im.size
            except Exception:
                if bio.tell() > 5 * 1024 * 1024:
                    break
        bio.seek(0)
        with Image.open(bio) as im:
            im.load()
            return im.size
    except Exception:
        return (0, 0)

def lumens_upgrade_scene7_url(url: str) -> str:
    if not isinstance(url, str) or "is/image" not in url:
        return url
    host = urlparse(url).netloc.lower()
    if "lumens" not in host and "scene7" not in host and "is/image" not in url:
        return url

    candidates = [url]
    if "?" in url:
        path, q = url.split("?", 1)
        if q.startswith("$") and q.endswith("$"):
            for preset in [
                "$Lumens.com-PDP-zoom$",
                "$Lumens.com-Product-Zoom$",
                "$Lumens.com-zoom$",
                "$zoom$",
            ]:
                candidates.append(f"{path}?{preset}")

    no_preset = _strip_scene7_preset(url)
    candidates.append(no_preset)

    for wid in (2400, 3000, 4000):
        for extras in (
            {"wid": wid, "qlt": 95, "fmt": "jpg"},
            {"wid": wid, "qlt": 95, "fmt": "jpg", "scl": 1},
            {"wid": wid, "qlt": 95, "fmt": "jpg", "scl": 1, "resMode": "sharp2", "op_usm": "1.0,1.0,6,0"},
        ):
            candidates.append(_update_query(no_preset, extras))
    candidates.append(_update_query(url, {"wid": 3000, "qlt": 95, "fmt": "jpg"}))

    seen, uniq = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    best_url, best_w, best_h = url, -1, -1
    for cand in uniq:
        w, h = _measure_image_dims(cand)
        if w > best_w or (w == best_w and h > best_h):
            best_url, best_w, best_h = cand, w, h
    return best_url

def scene7_upgrade_generic(url: str) -> str:
    if not isinstance(url, str) or "is/image" not in url:
        return url
    host = urlparse(url).netloc.lower()
    if not any(k in host for k in ("crateandbarrel", "crate", "cb2", "scene7")):
        return url

    base = url.split("?", 1)[0]
    candidates = [url, base]
    for wid in (2400, 3000, 4000):
        candidates.append(_update_query(base, {"wid": wid, "qlt": 95, "fmt": "jpg"}))
        candidates.append(_update_query(base, {"wid": wid, "qlt": 95, "fmt": "jpg", "scl": 1}))

    seen, uniq = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c); uniq.append(c)

    best, bw, bh = url, -1, -1
    for cand in uniq:
        w, h = _measure_image_dims(cand)
        if w > bw or (w == bw and h > bh):
            best, bw, bh = cand, w, h
    return best

# ===================== Firecrawl v2 (REST) helpers =====================
def firecrawl_scrape_v2(url: str, api_key: str, mode: str = "simple") -> dict:
    """Firecrawl /v2/scrape (simple or gentle)."""
    if not api_key:
        return {}
    payload = {
        "url": url,
        "formats": [
            "html",
            "markdown",  # sometimes useful
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
    """Aggressive: deeper waits/scrolls + hover + JS harvest of lazy/srcset images and price."""
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
        "timeout": 70000,
        "device": "desktop",
        "actions": [
            {"type": "wait", "milliseconds": 1200},
            {"type": "scroll", "y": 800},
            {"type": "wait", "milliseconds": 1200},
            {"type": "scroll", "y": 1600},
            {"type": "wait", "milliseconds": 1200},
            {"type": "waitForSelector",
             "selector": "[data-enlarged], img[data-enlarged], [data-testid*='ImageCarousel'], [class*='ImageCarousel'], picture source[srcset], img[srcset], img[data-src], img[data-srcset]",
             "timeout": 10000},
            {"type": "evaluate",
             "script": """
               (()=>{
                 const cand = document.querySelector('[data-enlarged], [data-testid*="ImageCarousel"] img, img[data-src], img[srcset]') ||
                               document.querySelector('img');
                 if (cand) {
                   const ev = new MouseEvent('mouseover', {bubbles:true, cancelable:true, view:window});
                   cand.dispatchEvent(ev);
                 }
               })();
             """},
            {"type": "wait", "milliseconds": 900},
            {"type": "evaluate",
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
                   const v = m.getAttribute("content"); if (v) out.images.push(v);
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
                   let cand = img.getAttribute("data-enlarged") || img.getAttribute("src") || img.getAttribute("data-src") || "";
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
             """}
        ]
    }
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v2/scrape",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=95
        )
        if r.status_code >= 400:
            return {}
        return r.json()
    except Exception:
        return {}

# ===================== Parsers =====================
def pick_image_and_price_bs4(html: str, base_url: str) -> Tuple[str, str]:
    """OG/Twitter → JSON-LD → first <img>; price via JSON-LD/meta/visible."""
    soup = BeautifulSoup(html or "", "lxml")
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
                        if isinstance(im, list) and im:
                            img_url = urljoin(base_url, im[0]); break
                        if isinstance(im, str) and im:
                            img_url = urljoin(base_url, im); break
                if img_url: break
            except Exception:
                pass
    if not img_url:
        anyimg = soup.find("img", src=True)
        if anyimg: img_url = urljoin(base_url, anyimg["src"])

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
        meta_price = soup.find("meta", attrs={"itemprop":"price"}) or \
                     soup.find("meta", attrs={"property":"product:price:amount"})
        if meta_price and meta_price.get("content"):
            val = meta_price["content"]
            price = f"${val}" if not str(val).startswith("$") else str(val)
    if not price:
        m = PRICE_RE.search(soup.get_text(" ", strip=True))
        if m: price = m.group(0)
    return img_url or "", price or ""

# ----- Lumens specific -----
LUMENS_PDP_RE = re.compile(
    r'https://images\.lumens\.com/is/image/Lumens/[A-Za-z0-9_/-]+?\?\$Lumens\.com-PDP-large\$', re.I
)
def _first_lumens_pdp_large_from_html(html: str) -> str:
    if not html: return ""
    m = LUMENS_PDP_RE.search(html)
    if m: return m.group(0)
    soup = BeautifulSoup(html, "lxml")
    for im in soup.find_all("img"):
        for attr in ("data-src", "data-original", "data-zoom-image", "data-large_image", "src"):
            v = im.get(attr)
            if isinstance(v, str) and "$Lumens.com-PDP-large$" in v:
                return v
        ss = im.get("srcset") or im.get("data-srcset")
        if isinstance(ss, str) and "$Lumens.com-PDP-large$" in ss:
            cand = best_from_srcset(ss)
            if cand: return cand
    for pict in soup.find_all("picture"):
        for src in pict.find_all("source"):
            ss = src.get("srcset") or src.get("data-srcset")
            if isinstance(ss, str) and "$Lumens.com-PDP-large$" in ss:
                cand = best_from_srcset(ss)
                if cand: return cand
    preload = soup.find("link", rel=lambda v: v and "preload" in v, attrs={"as": "image"})
    if preload and isinstance(preload.get("href"), str) and "$Lumens.com-PDP-large$" in preload["href"]:
        return preload["href"]
    return ""

def parse_image_and_price_lumens_from_v2(scrape: dict) -> Tuple[str, str]:
    if not scrape: 
        return "", ""
    data = scrape.get("data") or {}
    html = data.get("html") or ""
    md   = data.get("markdown") or ""
    img = ""
    if isinstance(md, str):
        m = LUMENS_PDP_RE.search(md)
        if m: img = m.group(0)
    if not img:
        img = _first_lumens_pdp_large_from_html(html)
    price = ""
    j = data.get("json")
    if isinstance(j, dict):
        content = j.get("content") if isinstance(j.get("content"), dict) else j
        if isinstance(content, dict):
            price = (content.get("price") or "").strip()
    if not price and html:
        soup = BeautifulSoup(html or "", "lxml")
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
            m = soup.find("meta", attrs={"itemprop": "price"}) or \
                soup.find("meta", attrs={"property": "product:price:amount"})
            if m and m.get("content"):
                val = m["content"]; price = val if str(val).startswith("$") else f"${val}"
        if not price:
            t = soup.get_text(" ", strip=True)
            m = PRICE_RE.search(t)
            if m: price = m.group(0)
    return img or "", price or ""

# ----- Wayfair: Next.js fallback -----
def parse_wayfair_next_data(html: str) -> Tuple[str, str]:
    """Extract hero image + price from Wayfair's Next.js data if present."""
    if not html:
        return "", ""
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("script", id="__NEXT_DATA__", type="application/json")
    if not tag or not tag.string:
        return "", ""
    try:
        data = json.loads(tag.string)
    except Exception:
        return "", ""

    def walk(obj):
        if isinstance(obj, dict):
            # image candidates
            for k in ("imageUrl", "imageURL", "image", "url", "src", "srcLarge", "zoomUrl", "galleryImageUrl"):
                v = obj.get(k)
                if isinstance(v, str) and v.startswith(("http://", "https://")):
                    yield ("image", v)
            # price candidates
            for k in ("price", "salePrice", "currentPrice", "displayPrice", "minPrice"):
                v = obj.get(k)
                if isinstance(v, (str, int, float)):
                    yield ("price", str(v))
            for v in obj.values():
                yield from walk(v)
        elif isinstance(obj, list):
            for it in obj:
                yield from walk(it)

    img, price = "", ""
    for kind, val in walk(data):
        if kind == "image" and not img:
            img = val
        elif kind == "price" and not price:
            m = PRICE_RE.search(val.replace("USD", "$"))
            price = m.group(0) if m else price or val
        if img and price:
            break
    return img, price

# ----- Wayfair specific -----
def parse_image_and_price_wayfair_v2(scrape: dict) -> Tuple[str, str]:
    if not scrape: return "", ""
    data = scrape.get("data") or {}
    html = data.get("html") or ""
    meta = data.get("metadata") or {}
    img = ""; price = ""
    soup = BeautifulSoup(html or "", "lxml") if html else None

    # 1) JSON-LD Product
    if soup:
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                obj = json.loads(tag.string or "")
            except Exception:
                continue
            arr = obj if isinstance(obj, list) else [obj]
            for o in arr:
                t = o.get("@type")
                is_product = t == "Product" or (isinstance(t, list) and "Product" in t)
                if is_product:
                    im = o.get("image")
                    if isinstance(im, list) and im: img = im[0]
                    elif isinstance(im, str) and im: img = im
                    offers = o.get("offers") or {}
                    if isinstance(offers, list): offers = offers[0] if offers else {}
                    p = offers.get("price") or (offers.get("priceSpecification") or {}).get("price")
                    if p and not price:
                        price = p if str(p).startswith("$") else f"${p}"
            if img and price: break

    # 1.5) NEW: Next.js blob fallback
    if (not img or not price) and html:
        i_nx, p_nx = parse_wayfair_next_data(html)
        if i_nx and not img: img = i_nx
        if p_nx and not price: price = p_nx

    # 2) Gallery lazy attrs / srcset
    if not img and soup:
        gallery = soup.select_one('[data-enlarged], [data-testid*="ImageCarousel"], [class*="ImageCarousel"], [class*="Gallery"]') or soup
        cands = []
        for imgtag in gallery.find_all("img"):
            cand = imgtag.get("data-enlarged") or imgtag.get("data-src") or imgtag.get("src")
            if not cand:
                ss = imgtag.get("srcset") or imgtag.get("data-srcset")
                if ss:
                    best = ""; bestw = -1
                    for part in ss.split(","):
                        s = part.strip()
                        if not s: continue
                        parts = s.split()
                        u = parts[0]; w = -1
                        if len(parts) > 1 and parts[1].endswith("w"):
                            try: w = int(parts[1][:-1])
                            except: w = -1
                        if w > bestw: bestw, best = w, u
                    cand = best or cand
            if cand:
                score = 0
                s = " ".join((imgtag.get("class") or [])).lower()
                if any(k in s for k in ("hero","zoom","primary")): score += 5
                if "wayfair" in cand.lower() or "secure.img" in cand.lower(): score += 3
                cands.append((score, cand))
        if cands:
            cands.sort(key=lambda x: x[0], reverse=True)
            img = cands[0][1]

    # 3) OG/Twitter last resort
    if not img:
        img = meta.get("og:image") or meta.get("twitter:image") or meta.get("image") or ""

    # 4) Price fallback
    if not price and soup:
        m = PRICE_RE.search(soup.get_text(" ", strip=True))
        if m: price = m.group(0)
    return img or "", price or ""

# ----- Crate & Barrel / CB2 specific -----
def parse_image_and_price_crate_cb2_v2(scrape: dict) -> Tuple[str, str]:
    if not scrape: return "", ""
    data = scrape.get("data") or {}
    html = data.get("html") or ""
    meta = data.get("metadata") or {}
    soup = BeautifulSoup(html or "", "lxml") if html else None

    img = meta.get("og:image") or meta.get("twitter:image") or ""
    price = ""

    if soup:
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                obj = json.loads(tag.string or "")
            except Exception:
                continue
            arr = obj if isinstance(obj, list) else [obj]
            for o in arr:
                t = o.get("@type")
                if t == "Product" or (isinstance(t, list) and "Product" in t):
                    im = o.get("image")
                    if not img:
                        if isinstance(im, list) and im: img = im[0]
                        elif isinstance(im, str) and im: img = im
                    offers = o.get("offers") or {}
                    if isinstance(offers, list): offers = offers[0] if offers else {}
                    p = offers.get("price") or (offers.get("priceSpecification") or {}).get("price")
                    if p and not price:
                        price = p if str(p).startswith("$") else f"${p}"
            if img and price: break

        if not img:
            for imgtag in soup.find_all("img"):
                for attr in ("data-src", "src", "data-zoom-image"):
                    v = imgtag.get(attr)
                    if isinstance(v, str) and "/is/image/" in v:
                        img = v; break
                if img: break

    if not price and soup:
        m = PRICE_RE.search(soup.get_text(" ", strip=True))
        if m: price = m.group(0)

    if img and "/is/image/" in img:
        img = scene7_upgrade_generic(img)

    return img or "", price or ""

# ===================== Domain enrichers (use Firecrawl then fallback) =====================
def enrich_wayfair_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    # cache-buster helps avoid stale/light variants
    busted = add_cache_buster(url)
    sc = firecrawl_scrape_v2_aggressive(busted, api_key)
    i1, p1 = parse_image_and_price_wayfair_v2(sc)
    if i1 or p1:
        return i1, p1, "firecrawl_v2_aggressive"
    sc2 = firecrawl_scrape_v2(busted, api_key, mode="gentle")
    i2, p2 = parse_image_and_price_wayfair_v2(sc2)
    return i2, p2, ("firecrawl_v2_gentle" if (i2 or p2) else "firecrawl_v2_fail")

def enrich_crate_cb2_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    sc = firecrawl_scrape_v2_aggressive(url, api_key)
    i1, p1 = parse_image_and_price_crate_cb2_v2(sc)
    if i1 or p1:
        return i1, p1, "firecrawl_v2_aggressive"
    sc2 = firecrawl_scrape_v2(url, api_key, mode="gentle")
    i2, p2 = parse_image_and_price_crate_cb2_v2(sc2)
    return i2, p2, ("firecrawl_v2_gentle" if (i2 or p2) else "firecrawl_v2_fail")

def enrich_ferguson_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    sc = firecrawl_scrape_v2(url, api_key, mode="simple")
    # generic parse works OK
    i1, p1 = pick_image_and_price_bs4(sc.get("data", {}).get("html", "") or "", url)
    if i1 or p1:
        return i1, p1, "firecrawl_v2_simple"
    sc2 = firecrawl_scrape_v2(url, api_key, mode="gentle")
    i2, p2 = pick_image_and_price_bs4(sc2.get("data", {}).get("html", "") or "", url)
    return i2, p2, ("firecrawl_v2_gentle" if (i2 or p2) else "firecrawl_v2_fail")

def enrich_lumens_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    u = canonicalize_url(url)
    sc = firecrawl_scrape_v2(u, api_key, mode="simple")
    img, price = parse_image_and_price_lumens_from_v2(sc)
    status = "firecrawl_v2_simple"
    if not img or not price:
        sc2 = firecrawl_scrape_v2(u, api_key, mode="gentle")
        i2, p2 = parse_image_and_price_lumens_from_v2(sc2)
        img = img or i2
        price = price or p2
        status = "firecrawl_v2_gentle" if (i2 or p2) else status
    if img and ("images.lumens.com/is/image" in img or "/is/image/" in img):
        img = lumens_upgrade_scene7_url(img)
    return img, price, status

def enrich_domain_firecrawl_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    sc = firecrawl_scrape_v2_aggressive(url, api_key)
    i1, p1 = pick_image_and_price_bs4(sc.get("data", {}).get("html", "") or "", url)
    if i1 or p1:
        return i1, p1, "firecrawl_v2_aggressive"
    sc2 = firecrawl_scrape_v2(url, api_key, mode="gentle")
    i2, p2 = pick_image_and_price_bs4(sc2.get("data", {}).get("html", "") or "", url)
    return i2, p2, ("firecrawl_v2_gentle" if (i2 or p2) else "firecrawl_v2_fail")

# ===================== Sidebar (API key) =====================
with st.sidebar:
    st.subheader("Firecrawl (optional)")
    api_key_input = st.text_input(
        "FIRECRAWL_API_KEY",
        value=os.getenv("FIRECRAWL_API_KEY", ""),
        type="password",
        help="Put this in Streamlit Cloud → Settings → Secrets, or paste it here."
    )
    st.caption("Leave blank to use the built-in parser only (no credits used).")

# ===================== Tabs =====================
tab1, tab2, tab3 = st.tabs([
    "1) Extract from PDF",
    "2) Enrich CSV (Image URL + Price)",
    "3) Test single URL"
])

# ===================== Batch enrich =====================
def enrich_urls(df: pd.DataFrame, url_col: str, api_key: Optional[str]) -> pd.DataFrame:
    """
    Add scraped_image_url and price; Firecrawl v2 first (domain-aware),
    fallback to bs4, with live progress bar + ETA.
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
        mask_done = pd.Series([False]*len(out), index=out.index)

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
            low = u.lower()

            # ---- Firecrawl v2 (domain-aware) ----
            if api_key:
                if "lumens.com" in low:
                    img, price, st_code = enrich_lumens_v2(u, api_key)
                elif "wayfair.com" in low:
                    img, price, st_code = enrich_wayfair_v2(u, api_key)
                elif "crateandbarrel.com" in low or "cb2.com" in low:
                    img, price, st_code = enrich_crate_cb2_v2(u, api_key)
                elif "fergusonhome.com" in low:
                    img, price, st_code = enrich_ferguson_v2(u, api_key)
                else:
                    img, price, st_code = enrich_domain_firecrawl_v2(u, api_key)

            # ---- Fallback: requests + BeautifulSoup ----
            if not img or not price:
                r = requests_get(u)
                if r and r.text:
                    i2, p2 = pick_image_and_price_bs4(r.text, u)
                    img = img or i2
                    price = price or p2
                    st_code = (st_code + "+bs4_ok") if st_code else "bs4_ok"
                else:
                    st_code = (st_code + "+fetch_failed") if st_code else "fetch_failed"

            # Final Scene7 upgrades
            if img and ("images.lumens.com/is/image" in img or "/is/image/" in img):
                if "lumens" in (urlparse(img).netloc.lower()):
                    img = lumens_upgrade_scene7_url(img)
                elif any(k in img for k in ("/is/image/Crate", "/is/image/CB2", "crateandbarrel", "cb2")):
                    img = scene7_upgrade_generic(img)

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

        # polite anti-throttle jitter
        time.sleep(0.30 + 0.50*random.random())

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
    st.caption("Paste a single product URL and test the enrichment (domain-aware Firecrawl, then fallback).")
    test_url = st.text_input(
        "Product URL to test",
        "https://www.wayfair.com/furniture/pdp/example"
    )

    if st.button("Run test", key="single_test_btn"):
        test_url = canonicalize_url(test_url)
        img = price = ""; status = ""
        low = test_url.lower()
        if api_key_input:
            if "lumens.com" in low:
                img, price, status = enrich_lumens_v2(test_url, api_key_input)
            elif "wayfair.com" in low:
                img, price, status = enrich_wayfair_v2(test_url, api_key_input)
            elif "crateandbarrel.com" in low or "cb2.com" in low:
                img, price, status = enrich_crate_cb2_v2(test_url, api_key_input)
            elif "fergusonhome.com" in low:
                img, price, status = enrich_ferguson_v2(test_url, api_key_input)
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

        # Final Scene7 upgrade if applicable
        if img and ("images.lumens.com/is/image" in img or "/is/image/" in img):
            if "lumens" in (urlparse(img).netloc.lower()):
                img = lumens_upgrade_scene7_url(img)
            elif any(k in img for k in ("/is/image/Crate", "/is/image/CB2", "crateandbarrel", "cb2")):
                img = scene7_upgrade_generic(img)

        st.write("**Status:**", status or "unknown")
        st.write("**Image URL:**", img or "—")
        st.write("**Price:**", price or "—")
        if img:
            st.image(img, caption="Preview", use_container_width=True)
