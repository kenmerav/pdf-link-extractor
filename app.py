import os, io, re, json, time, requests
from typing import Optional, Tuple
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

# ----------------- Streamlit setup -----------------
st.set_page_config(page_title="Spec Link Extractor & Enricher", layout="wide")
st.title("🧰 Spec Link Extractor & Enricher")

# --- Progress timer helper ---
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0

# ========================= Shared HTTP/parsing helpers =========================
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

# ========================= TAB 1: Canva PDF extractor =========================
# Robust parsers (case-insensitive, tolerate ":" or "-", smart quotes, etc.)
QTY_RE     = re.compile(r"(?i)\b(?:QTY|Quantity)\b\s*[:\-]?\s*([0-9X]+)\b")
FINISH_RE  = re.compile(r"(?i)\bFinish\b\s*[:\-]?\s*(.+)")
SIZE_RE    = re.compile(r"(?i)\b(?:Size|Dimensions?)\b\s*[:\-]?\s*(.+)")
TYPE_RE    = re.compile(r"(?i)\bType\b\s*[:\-]?\s*(.+)")

# bullets/positions: “1.”, “1 ”, middle dot, small dot
POS_AT_START = re.compile(r'^\s*(\d{1,3})[.\u2024\u00B7]?\s*')
NEXT_BULLET  = re.compile(r'\s(\d{1,3})[.\u2024\u00B7](?=\s|[A-Z])')

def _normalize_separators(s: str) -> str:
    if not s: return ""
    # unroll newlines into pipes, keep existing pipes/semicolons spaced
    s = s.replace("\n", " | ")
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\s*;\s*", "; ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def parse_link_title_fields(link_text: str) -> dict:
    fields = {"Type": "", "QTY": "", "Finish": "", "Size": ""}
    s = _normalize_separators(link_text)
    if not s:
        return fields

    # Split on pipes primarily; allow semicolons inside each chunk
    parts = []
    for chunk in s.split("|"):
        subs = [x.strip() for x in chunk.split(";")]
        parts.extend([x for x in subs if x])

    for tok in parts:
        m = TYPE_RE.search(tok)
        if m and not fields["Type"]:
            fields["Type"] = m.group(1).strip(); continue
        m = QTY_RE.search(tok)
        if m and not fields["QTY"]:
            q = m.group(1).strip().upper()
            fields["QTY"] = "" if q == "XX" else q
            continue
        m = FINISH_RE.search(tok)
        if m and not fields["Finish"]:
            fields["Finish"] = m.group(1).strip(); continue
        m = SIZE_RE.search(tok)
        if m and not fields["Size"]:
            size_val = m.group(1).strip()
            size_val = size_val.replace("”", '"').replace("“", '"').replace("’", "'").replace("‘", "'")
            size_val = re.sub(r"\s*[xX]\s*", " x ", size_val)
            fields["Size"] = size_val.strip(); continue

    # If Type was unlabeled (e.g., "PENDANTS | QTY: 2 | ..."), infer from first unlabeled token
    if not fields["Type"]:
        for tok in parts:
            if ":" in tok or "-" in tok:  # likely a labeled field
                continue
            if tok.strip():
                fields["Type"] = tok.strip()
                break

    return fields

def extract_link_title_strict(page, rect, pad_px: float = 4.0, band_px: float = 28.0) -> str:
    """
    STRICT per-link capture but expanded to the *entire text line* that the link token sits on.
    Why: In Canva, a hyperlink can be applied to just the bullet (e.g., "2.") or a single
    token in a line. If we only keep words whose centers lie inside the rect, we can end up
    with rows like "2." or "5" instead of the full line. This version:
      1) Collect words whose center lies in the slightly padded rect (R).
      2) If empty, look in a thin horizontal band that overlaps X with the rect.
      3) Expand to include *all words on the same (block,line)* as any kept word.
      4) As a final fallback, use page.get_textbox(rect).
    """
    import fitz
    if not rect:
        return ""
    r = fitz.Rect(rect).normalize()
    R = fitz.Rect(r.x0 - pad_px, r.y0 - pad_px, r.x1 + pad_px, r.y1 + pad_px)

    # words: [x0,y0,x1,y1,word,block,line,word_no]
    words = page.get_text("words") or []

    kept = []  # (y0, x0, word, block, line)
    for x0, y0, x1, y1, w, b, ln, *_ in words:
        if not w:
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if R.contains(fitz.Point(cx, cy)):
            kept.append((y0, x0, w, b, ln))

    if not kept:
        # band fallback: same Y ± band, with X overlap
        band = fitz.Rect(r.x0, r.y0 - band_px, r.x1, r.y1 + band_px)
        for x0, y0, x1, y1, w, b, ln, *_ in words:
            if not w:
                continue
            cy_in = band.y0 <= ((y0 + y1) / 2.0) <= band.y1
            x_overlaps = not (x1 < r.x0 or x0 > r.x1)
            if cy_in and x_overlaps:
                kept.append((y0, x0, w, b, ln))

    if kept:
        # Expand to the full (block,line) of the kept word(s). Prefer the line with the most words.
        line_keys = [(b, ln) for *_ignore, b, ln in kept]
        # Count words per line
        counts = {}
        for key in line_keys:
            counts[key] = counts.get(key, 0) + 1
        # Pick the most represented (block,line)
        best_key = max(counts.items(), key=lambda kv: kv[1])[0]
        bx, lx = best_key
        line_words = [(y0, x0, w) for x0, y0, x1, y1, w, b, ln, *_ in words if b == bx and ln == lx and w]
        line_words.sort(key=lambda t: (round(t[0], 3), t[1]))
        text = " ".join(t[2] for t in line_words).strip()
    else:
        text = ""

    if not text:
        # last resort: whatever the textbox returns
        try:
            text = (page.get_textbox(R) or "").strip()
        except Exception:
            text = ""

    return _normalize_separators(text)

def split_position_and_title_start(raw: str) -> tuple[str, str]:
    s = (raw or "").strip()
    if not s:
        return "", ""
    m = POS_AT_START.match(s)
    if not m:
        return "", s
    pos = m.group(1)
    rest = s[m.end():].strip()
    # If we accidentally captured the next bullet too, cut it off
    m2 = NEXT_BULLET.search(rest)
    if m2:
        rest = rest[:m2.start()].strip()
    return pos, rest

# ----------------- Room mapping logic -----------------
ROOM_MAP_RAW = [
    ("Sink", "Plumbing"),
    ("Faucet", "Plumbing"),
    ("Sink Faucet", "Plumbing"),
    ("Shower", "Plumbing"),
    ("Shower Head", "Plumbing"),
    ("Tub", "Plumbing"),
    ("Shower System", "Plumbing"),
    ("Shower Drain", "Plumbing"),
    ("Pendant", "Lighting"),
    ("Sconce", "Lighting"),
    ("Sconces", "Lighting"),
    ("Lamp", "Lighting"),
]

# build a lowercase lookup dict for exact/starts-with style checks
ROOM_MAP = {k.lower(): v for (k, v) in ROOM_MAP_RAW}
ROOM_OPTIONS = sorted(set(v for _, v in ROOM_MAP_RAW))
ROOM_OPTIONS = ["", *ROOM_OPTIONS, "Other"]

def _infer_room_from_tag(tag_val: str) -> str:
    """
    Given a 'Tags' value from the page (ex: "Sink" / "Pendant" / etc.),
    return the mapped Room ("Plumbing", "Lighting", ...).

    Strategy:
    1. exact lowercase match
    2. startswith match (so "Sink Faucet" still hits "Sink Faucet" first, then "Sink")
    Fallback: "" (blank)
    """
    if not tag_val:
        return ""
    t = tag_val.strip().lower()

    # exact
    if t in ROOM_MAP:
        return ROOM_MAP[t]

    # try longest key that is prefix of t
    best_key = ""
    for k in ROOM_MAP.keys():
        if t.startswith(k) and len(k) > len(best_key):
            best_key = k
    return ROOM_MAP.get(best_key, "")

def extract_links_by_pages(
    pdf_bytes: bytes,
    page_to_tag: dict[int, str] | None,
    page_to_room: dict[int, str] | None = None,
    only_listed_pages: bool = True,
    pad_px: float = 4.0,
    band_px: float = 28.0,
) -> pd.DataFrame:
    doc = fitz.open("pdf", pdf_bytes)
    rows = []
    listed = set(page_to_tag.keys()) if page_to_tag else set()

    for pidx, page in enumerate(doc, start=1):
        if only_listed_pages and page_to_tag and pidx not in listed:
            continue
        tag_value = (page_to_tag or {}).get(pidx, "")
        room_value = (page_to_room or {}).get(pidx, _infer_room_from_tag(tag_value))

        for lnk in page.get_links():
            uri = (lnk.get("uri") or "").strip()
            if not uri.lower().startswith(("http://", "https://")):
                continue

            rect = lnk.get("from")
            raw = extract_link_title_strict(page, rect, pad_px=pad_px, band_px=band_px)
            position, title = split_position_and_title_start(raw)
            # Ignore common headings like "MATERIALS LIST"
            if not title or title.strip().lower().startswith(("materials list","material list")):
                continue
            fields = parse_link_title_fields(title)

            rows.append({
                "page": pidx,
                "Tags": tag_value,                "Room": room_value,
                "Position": position,
                "Type": fields.get("Type", ""),
                "QTY": fields.get("QTY", ""),
                "Finish": fields.get("Finish", ""),
                "Size": fields.get("Size", ""),
                "link_url": uri,
                "link_text": title,
            })
    return pd.DataFrame(rows)

# ========================= Tabs 2/3: Your Firecrawl + parsers =========================
def pick_image_and_price_bs4(html: str, base_url: str) -> Tuple[str, str]:
    """Lightweight fallback: og/twitter → JSON-LD → meta → visible $ pattern."""
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

# --------- Lumens-specific helpers (targets PDP-large + lazyload) ----------
LUMENS_PDP_RE = re.compile(
    r'https://images\.lumens\.com/is/image/Lumens/[A-Za-z0-9_/-]+?\?\$Lumens\.com-PDP-large\$', re.I
)

def _largest_from_srcset(srcset_value: str) -> str:
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
            cand = _largest_from_srcset(ss)
            if cand: return cand

    for pict in soup.find_all("picture"):
        for src in pict.find_all("source"):
            ss = src.get("srcset") or src.get("data-srcset")
            if isinstance(ss, str) and "$Lumens.com-PDP-large$" in ss:
                cand = _largest_from_srcset(ss)
                if cand: return cand

    preload = soup.find("link", rel=lambda v: v and "preload" in v, attrs={"as": "image"})
    if preload and isinstance(preload.get("href"), str) and "$Lumens.com-PDP-large$" in preload["href"]:
        return preload["href"]

    return ""

def parse_image_and_price_lumens_from_v2(scrape: dict) -> Tuple[str, str]:
    """Lumens: prefer PDP-large in markdown/html, then meta/JSON-LD/visible."""
    if not scrape:
        return "", ""
    data = scrape.get("data") or {}
    html = data.get("html") or ""
    md   = data.get("markdown") or ""

    img = ""
    if isinstance(md, str):
        m = LUMENS_PDP_RE.search(md)
        if m:
            img = m.group(0)
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
                val = m["content"]
                price = val if str(val).startswith("$") else f"${val}"
        if not price:
            t = soup.get_text(" ", strip=True)
            m = PRICE_RE.search(t)
            if m: price = m.group(0)

    return img or "", price or ""

# ----------------- Firecrawl v2 (REST) helpers -----------------
def firecrawl_scrape_v2(url: str, api_key: str, mode: str = "simple") -> dict:
    """
    Call Firecrawl /v2/scrape via REST.
    We ask for HTML + MARKDOWN so we can regex the PDP-large image.
    """
    if not api_key:
        return {}
    payload = {
        "url": url,
        "formats": [
            "html",
            "markdown",
            { "type": "json", "schema": {
                "type": "object",
                "properties": { "price": { "type": "string" } },
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

def parse_image_and_price_from_v2_generic(scrape: dict) -> Tuple[str, str]:
    """Generic Firecrawl parse: OG/Twitter/meta + JSON-LD + visible."""
    if not scrape: return "", ""
    data = scrape.get("data") or {}
    meta = data.get("metadata") or {}
    html = data.get("html") or ""

    img = meta.get("og:image") or meta.get("twitter:image") or meta.get("image") or ""
    price = ""
    if html:
        soup = BeautifulSoup(html or "", "lxml")
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
            m = PRICE_RE.search(soup.get_text(" ", strip=True))
            if m: price = m.group(0)
    return img or "", price or ""

def enrich_domain_firecrawl_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    sc = firecrawl_scrape_v2(url, api_key, mode="simple")
    img, price = parse_image_and_price_from_v2_generic(sc)
    status = "firecrawl_v2_simple"
    if not img or not price:
        sc2 = firecrawl_scrape_v2(url, api_key, mode="gentle")
        i2, p2 = parse_image_and_price_from_v2_generic(sc2)
        img = img or i2
        price = price or p2
        status = "firecrawl_v2_gentle" if (i2 or p2) else status
    return img, price, status

def enrich_wayfair_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    # (From your version: reuses generic flow)
    return enrich_domain_firecrawl_v2(url, api_key)

def enrich_ferguson_v2(url: str, api_key: str) -> Tuple[str, str, str]:
    return enrich_domain_firecrawl_v2(url, api_key)

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
    return img, price, status

# ----------------- Sidebar (API key) -----------------
with st.sidebar:
    st.subheader("Firecrawl (optional)")
    api_key_input = st.text_input(
        "FIRECRAWL_API_KEY",
        value=os.getenv("FIRECRAWL_API_KEY", ""),
        type="password",
        help="Put this in Streamlit Cloud → Settings → Secrets, or paste it here."
    )
    st.caption("Leave blank to use the built-in parser only (no credits used).")

# ----------------- Tabs -----------------
tab1, tab2, tab3 = st.tabs([
    "1) Extract from PDF (pages + Tags + Position + full titles)",
    "2) Enrich CSV (Image URL + Price)",
    "3) Test single URL"
])

# --- Tab 1: Canva PDF → rows ---
with tab1:
    st.caption("Build a page→Tags table, then extract ALL web links on those pages. Each link stays on its own row.")
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
    pad_px = st.slider("Link capture pad (pixels)", 0, 16, 4, 1)
    band_px = st.slider("Nearby text band (pixels)", 0, 60, 28, 2)

    run1 = st.button("Extract", type="primary", disabled=(pdf_file is None), key="extract_btn")
    if run1 and pdf_file:
        page_to_tag = {}
        if num_pages is None:
            try:
                num_pages = len(fitz.open("pdf", pdf_file.getvalue()))
            except:
                num_pages = None
        for _, row in mapping_df.iterrows():
            p_raw = str(row.get("page","")); p_raw = p_raw.strip()
            t_raw = str(row.get("Tags","")); t_raw = t_raw.strip()
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
                pdf_bytes, page_to_tag, None,
                only_listed_pages=only_listed,
                pad_px=pad_px,
                band_px=band_px
            )
        if df.empty:
            st.info("No links found. Verify the PDF uses live hyperlinks (not just images).")
        else:
            st.success(f"Extracted {len(df)} row(s).")
            st.caption("Edit the Room per row if needed, then download your CSV.")
            edited_df = st.data_editor(
                df,
                use_container_width=True,
                column_config={
                    "Room": st.column_config.SelectboxColumn("Room", options=ROOM_OPTIONS, help="Choose a room/category or leave blank")
                }
            )
            st.download_button(
                "Download CSV",
                edited_df.to_csv(index=False).encode("utf-8"),
                file_name="canva_links_with_position.csv",
                mime="text/csv"
            )

# --- Tab 2: Enrich CSV (your version preserved) ---
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

            def enrich_urls(df: pd.DataFrame, url_col: str, api_key: Optional[str]) -> pd.DataFrame:
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
                        if api_key:
                            if "lumens.com" in u:
                                img, price, st_code = enrich_lumens_v2(u, api_key)
                            elif "fergusonhome.com" in u:
                                img, price, st_code = enrich_ferguson_v2(u, api_key)
                            elif "wayfair.com" in u:
                                img, price, st_code = enrich_wayfair_v2(u, api_key)
                            else:
                                img, price, st_code = enrich_domain_firecrawl_v2(u, api_key)

                        if not img or not price:
                            r = requests_get(u)
                            if r and r.text:
                                i2, p2 = pick_image_and_price_bs4(r.text, u)
                                img = img or i2
                                price = price or p2
                                st_code = (st_code + "+bs4_ok") if st_code else "bs4_ok"
                            else:
                                st_code = (st_code + "+fetch_failed") if st_code else "fetch_failed"

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

# --- Tab 3: Test a single URL (your version preserved) ---
with tab3:
    st.caption("Paste a single product URL and test the enrichment (Firecrawl v2 first, then fallback).")
    test_url = st.text_input(
        "Product URL to test",
        "https://www.lumens.com/vishal-chandelier-by-troy-lighting-TRY2622687.html?utm_source=google&utm_medium=PLA&utm_brand=Troy-Lighting&utm_id=TRY2622687&utm_campaign=189692751"
    )
    if st.button("Run test", key="single_test_btn"):
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

        st.write("**Status:**", status or "unknown")
        st.write("**Image URL:**", img or "—")
        st.write("**Price:**", price or "—")
        if img:
            st.image(img, caption="Preview", use_container_width=True)

