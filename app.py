# app.py — Firecrawl-hybrid (render=True + aggressive + domain parsers)
# Tabs:
# 1) Extract links from PDF (simple extractor)
# 2) Enrich CSV (Firecrawl v2 simple→gentle→aggressive, then domain parsers, then generic bs4)
# 3) Test a single URL

import os, io, json, re, time, random, requests
from typing import Optional, Tuple
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===================== Streamlit setup =====================
st.set_page_config(page_title="Spec Link Extractor & Enricher (Firecrawl-hybrid)", layout="wide")
st.title("🧰 Spec Link Extractor & Enricher — Firecrawl Hybrid")

# ===================== Helpers =====================
PRICE_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
UA = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def requests_get(url: str, timeout: int = 25, retries: int = 2) -> Optional[requests.Response]:
    for _ in range(retries+1):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if 200 <= r.status_code < 300:
                return r
        except Exception:
            pass
        time.sleep(0.35 + 0.3*random.random())
    return None

def canonicalize_url(u: str) -> str:
    """Remove tracking params; keep canonical path."""
    try:
        p = urlparse(u)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith(("utm_", "gclid", "gbraid", "wbraid", "msclkid", "mc_"))]
        return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), ""))
    except Exception:
        return u

def add_cache_buster(u: str) -> str:
    """Add a throwaway param so Firecrawl won't return a cached render."""
    try:
        p = urlparse(u)
        q = dict(parse_qsl(p.query, keep_blank_values=True))
        q["_fcv"] = str(int(time.time()*1000))
        return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), ""))
    except Exception:
        return u

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

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0

# ===================== PDF → Links extractor (simple) =====================
def extract_links_from_pdf(pdf_bytes: bytes) -> pd.DataFrame:
    """Extract links + simple titles from uploaded PDF."""
    doc = fitz.open("pdf", pdf_bytes)
    rows = []
    for page in doc:
        for lnk in page.get_links():
            uri = lnk.get('uri') or ''
            if not uri.startswith(('http://', 'https://')):
                continue
            rect = lnk.get('from')
            title = ""
            if rect:
                try:
                    title = page.get_textbox(fitz.Rect(rect)).strip()
                except Exception:
                    title = ""
            rows.append({
                'Product Name': title,
                'Product URL': uri,
            })
    return pd.DataFrame(rows)

# ===================== Scene7 helpers (Lumens/Crate/CB2) =====================
def _update_query(url: str, add: dict, remove_keys: tuple = ()) -> str:
    p = urlparse(url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    for k in remove_keys:
        q.pop(k, None)
    q.update({k: str(v) for k, v in add.items()})
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), ""))

def _strip_scene7_preset(url: str) -> str:
    if "?" not in url: return url
    path, q = url.split("?", 1)
    q = q.strip()
    # remove $Preset$ tokens if present
    if q.startswith("$") and q.endswith("$"):
        return path
    if "$" in q:
        parts = [p for p in q.split("&") if not (p.startswith("$") and p.endswith("$"))]
        q = "&".join(parts)
        return f"{path}?{q}" if q else path
    return url

def _measure_image_dims(url: str, timeout: int = 10) -> tuple[int, int]:
    """Best-effort: stream some bytes and let Pillow read size; bail if too big."""
    try:
        r = requests.get(url, headers=UA, stream=True, timeout=timeout)
        r.raise_for_status()
        bio = io.BytesIO()
        for chunk in r.iter_content(8192):
            if not chunk: break
            bio.write(chunk)
            try:
                bio.seek(0)
                with Image.open(bio) as im:
                    im.load()
                    return im.size
            except Exception:
                if bio.tell() > 4 * 1024 * 1024:  # stop early on huge images
                    break
        bio.seek(0)
        with Image.open(bio) as im:
            im.load(); return im.size
    except Exception:
        return (0, 0)

def lumens_upgrade_scene7_url(url: str) -> str:
    """Try to upsize Lumens Scene7 images to a large, clean variant."""
    if not isinstance(url, str) or "is/image" not in url or "lumens" not in url.lower():
        return url
    candidates = [url]
    no_preset = _strip_scene7_preset(url)
    candidates.append(no_preset)
    for wid in (2400, 3000, 4000):
        candidates.append(_update_query(no_preset, {"wid": wid, "qlt": 95, "fmt": "jpg"}))
        candidates.append(_update_query(no_preset, {"wid": wid, "qlt": 95, "fmt": "jpg", "scl": 1}))
    best_url, best_w, best_h = url, -1, -1
    for cand in dict.fromkeys(candidates):  # de-dupe
        w, h = _measure_image_dims(cand)
        if w > best_w or (w == best_w and h > best_h):
            best_url, best_w, best_h = cand, w, h
    return best_url

def scene7_upgrade_generic(url: str) -> str:
    """Upsize Crate/CB2 Scene7 images."""
    if not isinstance(url, str) or "is/image" not in url:
        return url
    if not any(k in url.lower() for k in ("crateandbarrel", "crate", "cb2", "scene7")):
        return url
    base = url.split("?", 1)[0]
    candidates = [url, base]
    for wid in (2400, 3000, 4000):
        candidates.append(_update_query(base, {"wid": wid, "qlt": 95, "fmt": "jpg"}))
        candidates.append(_update_query(base, {"wid": wid, "qlt": 95, "fmt": "jpg", "scl": 1}))
    best, bw, bh = url, -1, -1
    for cand in dict.fromkeys(candidates):
        w, h = _measure_image_dims(cand)
        if w > bw or (w == bw and h > bh):
            best, bw, bh = cand, w, h
    return best

# ===================== Pure-Requests Parsers =====================
def pick_image_and_price_bs4(html: str, base_url: str) -> Tuple[str, str]:
    """Generic: OG/Twitter → JSON-LD → first <img>; price via JSON-LD/meta/visible."""
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

# ----- Lumens specific (PDP-large pattern + lazy) -----
LUMENS_PDP_RE = re.compile(
    r'https://images\.lumens\.com/is/image/Lumens/[A-Za-z0-9_/-]+?\?\$Lumens\.com-PDP-large\$', re.I
)

def parse_image_and_price_lumens(html: str, base_url: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html or "", "lxml")
    text = html or ""
    img = ""
    m = LUMENS_PDP_RE.search(text)
    if m: img = m.group(0)
    if not img:
        for im in soup.find_all("img"):
            for attr in ("data-src", "data-original", "data-zoom-image", "data-large_image", "src"):
                v = im.get(attr)
                if isinstance(v, str) and "$Lumens.com-PDP-large$" in v:
                    img = v; break
            if img: break
        if not img:
            for src in soup.select("picture source[srcset], source[srcset]"):
                ss = src.get("srcset") or src.get("data-srcset")
                if isinstance(ss, str) and "$Lumens.com-PDP-large$" in ss:
                    cand = best_from_srcset(ss)
                    if cand: img = cand; break
    price = ""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        objs = data if isinstance(data, list) else [data]
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
        meta_price = soup.find("meta", attrs={"itemprop":"price"}) or soup.find("meta", attrs={"property": "product:price:amount"})
        if meta_price and meta_price.get("content"):
            val = meta_price["content"]
            price = val if str(val).startswith("$") else f"${val}"
    if not price:
        m = PRICE_RE.search(soup.get_text(" ", strip=True))
        if m: price = m.group(0)
    if img and "images.lumens.com/is/image" in img:
        img = lumens_upgrade_scene7_url(img)
    return img or "", price or ""

# ----- Wayfair specific: __NEXT_DATA__ + wfcdn fallback -----
def parse_wayfair_next_data(html: str) -> Tuple[str, str]:
    if not html: return "", ""
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("script", id="__NEXT_DATA__", type="application/json")
    if not tag or not tag.string: return "", ""
    try:
        data = json.loads(tag.string)
    except Exception:
        return "", ""
    img = ""
    price = ""
    # Heuristics: search for any object with wfcdn image and price-ish fields
    def walk(o):
        if isinstance(o, dict):
            yield o
            for v in o.values(): yield from walk(v)
        elif isinstance(o, list):
            for v in o: yield from walk(v)
    for node in walk(data):
        # image signals
        for k in ("image", "imageUrl", "src", "url"):
            v = node.get(k)
            if isinstance(v, str) and ("wfcdn" in v or v.endswith(".jpg") or v.endswith(".png")):
                img = img or v
        # price signals
        for k in ("price", "salePrice", "currentPrice", "displayPrice", "minPrice", "maxPrice"):
            v = node.get(k)
            if isinstance(v, (str, int, float)):
                s = str(v)
                m = PRICE_RE.search(s.replace("USD", "$"))
                if m:
                    price = price or m.group(0)
        if img and price:
            break
    return img, price

def parse_image_and_price_wayfair(html: str, base_url: str) -> Tuple[str, str]:
    img = price = ""
    i_nx, p_nx = parse_wayfair_next_data(html)
    if i_nx: img = i_nx
    if p_nx: price = p_nx
    if not img or not price:
        soup = BeautifulSoup(html or "", "lxml")
        if not img and soup:
            best, bestw = "", -1
            for im in soup.find_all("img"):
                v = im.get("src") or im.get("data-src") or ""
                if isinstance(v, str) and "wfcdn" in v:
                    best = v; bestw = max(bestw, 9999)
                ss = im.get("srcset") or im.get("data-srcset")
                if isinstance(ss, str) and "wfcdn" in ss:
                    for part in ss.split(","):
                        s = part.strip()
                        if not s: continue
                        parts = s.split()
                        u = parts[0]; w = -1
                        if len(parts) > 1 and parts[1].endswith("w"):
                            try: w = int(parts[1][:-1])
                            except: w = -1
                        if w > bestw:
                            bestw, best = w, u
            if best: img = best
        if not price and soup:
            m = PRICE_RE.search(soup.get_text(" ", strip=True))
            if m: price = m.group(0)
    return img or "", price or ""

# ----- Crate & Barrel / CB2 -----
def parse_image_and_price_crate_cb2(html: str, base_url: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html or "", "lxml")
    img = ""
    price = ""
    m = soup.find("meta", attrs={"property":"og:image"}) or soup.find("meta", attrs={"name":"og:image"}) if soup else None
    if m and m.get("content"): img = m["content"]
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
    if soup and not img:
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

# ===================== Firecrawl v2 (render=True) =====================
def firecrawl_scrape_v2(url: str, api_key: str, mode: str = "simple") -> dict:
    """Call Firecrawl /v2/scrape with render=True."""
    if not api_key:
        return {}
    payload = {
        "url": url,
        "formats": [
            "html",
            "markdown",
            {"type": "json", "schema": {
                "type": "object",
                "properties": {"price": {"type": "string"}},
                "required": []
            }}
        ],
        "proxy": "auto",
        "timeout": 45000,
        "device": "desktop",
        "render": True
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
    """Aggressive render with waits + mouseover + srcset evaluation."""
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
        "render": True,
        "actions": [
            {"type": "wait", "milliseconds": 1200},
            {"type": "scroll", "y": 800},
            {"type": "wait", "milliseconds": 1200},
            {"type": "scroll", "y": 1600},
            {"type": "wait", "milliseconds": 1200},
            {"type": "waitForSelector",
             "selector": "[data-enlarged], [data-testid*='ImageCarousel'], [data-testid*='media'], [data-automation*='ProductImage'], picture source[srcset], img[srcset], img[data-src], img[data-srcset]",
             "timeout": 12000},
            {"type": "evaluate",
             "script": """
               (()=>{
                 const sel = '[data-enlarged], [data-testid*="ImageCarousel"] img, [data-testid*="media"] img, [data-automation*="ProductImage"] img, img[data-src], img[srcset], img';
                 const cand = document.querySelector(sel);
                 if (cand) {
                   cand.dispatchEvent(new MouseEvent('mouseover', {bubbles:true, cancelable:true, view:window}));
                   cand.scrollIntoView({block:'center', inline:'center'});
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
                 const scope = document.querySelector('[data-testid*="ImageCarousel"], [class*="ImageCarousel"], [data-testid*="media"], [data-automation*="ProductImage"]') || document;
                 scope.querySelectorAll("picture source[srcset], source[srcset]").forEach(src => {
                   const best = bestFromSrcset(src.getAttribute("srcset"));
                   if (best) out.images.push(best);
                 });
                 scope.querySelectorAll("img").forEach(img => {
                   let cand = img.getAttribute("data-enlarged") || img.getAttribute("data-zoom-image") || img.getAttribute("data-large_image") ||
                              img.getAttribute("data-src") || img.getAttribute("src") || "";
                   if (!cand) {
                     const ss = img.getAttribute("srcset") || img.getAttribute("data-srcset");
                     if (ss) cand = bestFromSrcset(ss);
                   }
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

def parse_image_and_price_from_v2_generic(sc: dict) -> Tuple[str, str, str]:
    """Generic parse of Firecrawl result → (img, price, html)."""
    if not sc: return "", "", ""
    data = sc.get("data") or {}
    meta = data.get("metadata") or {}
    html = data.get("html") or ""
    img = meta.get("og:image") or meta.get("og:image:url") or meta.get("twitter:image") or meta.get("image") or ""
    price = ""
    j = data.get("json")
    if isinstance(j, dict):
        content = j.get("content") if isinstance(j.get("content"), dict) else j
        if isinstance(content, dict):
            price = (content.get("price") or "").strip()
    if html:
        soup = BeautifulSoup(html or "", "lxml")
        if not price:
            for tag in soup.find_all("script", type="application/ld+json"):
                try:
                    obj = json.loads(tag.string or "")
                except Exception:
                    continue
                arr = obj if isinstance(obj, list) else [obj]
                for o in arr:
                    t = o.get("@type")
                    if t == "Product" or (isinstance(t, list) and "Product" in t):
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

# ===================== Enrichment orchestrators =====================
def enrich_one_url_with_firecrawl(url: str, api_key: Optional[str]) -> Tuple[str, str, str]:
    """Firecrawl simple→gentle→aggressive; then domain parse on HTML; then local bs4."""
    u0 = canonicalize_url(url.strip())
    u = add_cache_buster(u0) if api_key else u0

    img = price = status = ""
    html = ""

    if api_key:
        sc1 = firecrawl_scrape_v2(u, api_key, mode="simple")
        i1, p1, h1 = parse_image_and_price_from_v2_generic(sc1)
        img = img or i1; price = price or p1; html = html or h1; status = "fc_simple"

        if not img or not price:
            sc2 = firecrawl_scrape_v2(u, api_key, mode="gentle")
            i2, p2, h2 = parse_image_and_price_from_v2_generic(sc2)
            img = img or i2; price = price or p2; html = html or h2; status = "fc_gentle" if (i2 or p2) else status

        if not img or not price:
            sc3 = firecrawl_scrape_v2_aggressive(u, api_key)
            i3, p3, h3 = parse_image_and_price_from_v2_generic(sc3)
            img = img or i3; price = price or p3; html = html or h3; status = "fc_aggressive" if (i3 or p3) else (status or "fc_none")

        # Run domain-specific pass on Firecrawl HTML (if we have it)
        if html:
            low = u0.lower()
            if "lumens.com" in low:
                i4, p4 = parse_image_and_price_lumens(html, u0)
                if i4: img = i4
                if p4: price = p4
                status += "+lumens_html"
            elif "wayfair.com" in low:
                i4, p4 = parse_image_and_price_wayfair(html, u0)
                if i4: img = i4
                if p4: price = p4
                status += "+wayfair_html"
            elif "crateandbarrel.com" in low or "cb2.com" in low:
                i4, p4 = parse_image_and_price_crate_cb2(html, u0)
                if i4: img = i4
                if p4: price = p4
                status += "+cratecb2_html"

    # Final fallback: direct fetch and parse
    if not img or not price:
        r = requests_get(u0)
        if r and r.text:
            low = u0.lower()
            if "lumens.com" in low:
                i5, p5 = parse_image_and_price_lumens(r.text, u0); status = (status + "+lumens_bs4") if status else "lumens_bs4"
            elif "wayfair.com" in low:
                i5, p5 = parse_image_and_price_wayfair(r.text, u0); status = (status + "+wayfair_bs4") if status else "wayfair_bs4"
            elif "crateandbarrel.com" in low or "cb2.com" in low:
                i5, p5 = parse_image_and_price_crate_cb2(r.text, u0); status = (status + "+cratecb2_bs4") if status else "cratecb2_bs4"
            else:
                i5, p5 = pick_image_and_price_bs4(r.text, u0); status = (status + "+generic_bs4") if status else "generic_bs4"
            img = img or i5; price = price or p5
        else:
            status = (status + "+fetch_failed") if status else "fetch_failed"

    return img or "", price or "", status or "unknown"

def enrich_one_url_no_fc(url: str) -> Tuple[str, str, str]:
    """Pure local fallback if no API key present."""
    u = canonicalize_url(url.strip())
    r = requests_get(u)
    if not r or not r.text:
        return "", "", "fetch_failed"
    html = r.text; low = u.lower()
    if "lumens.com" in low:
        img, price = parse_image_and_price_lumens(html, u); return img, price, "lumens_bs4"
    if "wayfair.com" in low:
        img, price = parse_image_and_price_wayfair(html, u); return img, price, "wayfair_bs4"
    if "crateandbarrel.com" in low or "cb2.com" in low:
        img, price = parse_image_and_price_crate_cb2(html, u); return img, price, "cratecb2_bs4"
    img, price = pick_image_and_price_bs4(html, u); return img, price, "generic_bs4"

# ===================== Sidebar (API key) =====================
with st.sidebar:
    st.subheader("Firecrawl (optional)")
    api_key_input = st.text_input(
        "FIRECRAWL_API_KEY",
        value=os.getenv("FIRECRAWL_API_KEY", ""),
        type="password",
        help="Paste your Firecrawl API key. Leave blank to use local-only fallback."
    )
    st.caption("Flow: Firecrawl render simple→gentle→aggressive, then domain parsers (Lumens/Wayfair/Crate/CB2).")

# ===================== Tabs =====================
tab1, tab2, tab3 = st.tabs([
    "1) Extract from PDF",
    "2) Enrich CSV (Image URL + Price)",
    "3) Test single URL"
])

# ===================== Tab 1: PDF → Links =====================
with tab1:
    st.caption("Upload a PDF → extract all web links → download a CSV.")
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

    prog = st.progress(0)
    status_box = st.empty()
    t_start = time.perf_counter()
    per_link_times = []
    key = (api_key or "").strip()

    for k, u in enumerate(urls, start=1):
        u = u.strip()
        img = price = ""; st_code = ""
        with Timer() as t:
            if key:
                img, price, st_code = enrich_one_url_with_firecrawl(u, key)
            else:
                img, price, st_code = enrich_one_url_no_fc(u)

            if not img or not price:
                # one more tiny try: direct generic bs4
                r = requests_get(u)
                if r and r.text:
                    i2, p2 = pick_image_and_price_bs4(r.text, u)
                    img = img or i2
                    price = price or p2
                    st_code = (st_code + "+bs4_ok") if st_code else "bs4_ok"
                else:
                    st_code = (st_code + "+fetch_failed") if st_code else "fetch_failed"

        imgs[k-1] = img; prices[k-1] = price; status[k-1] = st_code

        per_link_times.append(t.dt)
        avg = sum(per_link_times) / max(len(per_link_times), 1)
        remaining = (len(urls) - k) * avg
        status_box.write(
            f"Processed {k}/{len(urls)} • last {t.dt:.2f}s • avg {avg:.2f}s/link • ETA ~{int(remaining)}s"
        )
        prog.progress(k/len(urls))
        time.sleep(0.25 + 0.35*random.random())  # polite jitter

    total = time.perf_counter() - t_start
    if urls:
        status_box.write(f"Done {len(urls)} link(s) in {total:.1f}s • avg {(total/len(urls)):.2f}s/link")
    else:
        status_box.write("Nothing to do — no URLs found.")

    out["scraped_image_url"] = imgs
    out["price"] = prices
    out["scrape_status"] = status
    return out

with tab2:
    st.caption("Provide a CSV with a 'Product URL' column (or choose another).")
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
                with st.spinner("Scraping image + price…"):
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

# ===================== Tab 3: Test a single URL =====================
with tab3:
    st.caption("Paste a single product URL and test the enrichment (Firecrawl → domain parsers → fallback).")
    test_url = st.text_input(
        "Product URL to test",
        "https://www.lumens.com/xavier-wall-sconce-by-crystorama-CRYP535322.html"
    )
    if st.button("Run test", key="single_test_btn"):
        if api_key_input.strip():
            img, price, status = enrich_one_url_with_firecrawl(test_url, api_key_input.strip())
        else:
            img, price, status = enrich_one_url_no_fc(test_url)

        st.write("**Status:**", status or "unknown")
        st.write("**Image URL:**", img or "—")
        st.write("**Price:**", price or "—")
        if img:
            st.image(img, caption="Preview", use_container_width=True)
