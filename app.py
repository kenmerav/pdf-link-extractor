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
QTY_RE     = re.compile(r"(?i)\\b(?:QTY|Quantity)\\b\\s*[:\\-]?\\s*([0-9X]+)\\b")
FINISH_RE  = re.compile(r"(?i)\\bFinish\\b\\s*[:\\-]?\\s*(.+)")
SIZE_RE    = re.compile(r"(?i)\\b(?:Size|Dimensions?)\\b\\s*[:\\-]?\\s*(.+)")
TYPE_RE    = re.compile(r"(?i)\\bType\\b\\s*[:\\-]?\\s*(.+)")

POS_AT_START = re.compile(r'^\\s*(\\d{1,3})[.\\u2024\\u00B7]?\\s*')
NEXT_BULLET  = re.compile(r'\\s(\\d{1,3})[.\\u2024\\u00B7](?=\\s|[A-Z])')

def _normalize_separators(s: str) -> str:
    if not s: return ""
    s = s.replace("\n", " | ")
    s = re.sub(r"\\s*\\|\\s*", " | ", s)
    s = re.sub(r"\\s*;\\s*", "; ", s)
    s = re.sub(r"\\s{2,}", " ", s)
    return s.strip()

def parse_link_title_fields(link_text: str) -> dict:
    fields = {"Type": "", "QTY": "", "Finish": "", "Size": ""}
    s = _normalize_separators(link_text)
    if not s:
        return fields
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
            fields["QTY"] = "" if q == "XX" else q; continue
        m = FINISH_RE.search(tok)
        if m and not fields["Finish"]:
            fields["Finish"] = m.group(1).strip(); continue
        m = SIZE_RE.search(tok)
        if m and not fields["Size"]:
            size_val = m.group(1).strip()
            size_val = size_val.replace("”", '"').replace("“", '"').replace("’", "'").replace("‘", "'")
            size_val = re.sub(r"\\s*[xX]\\s*", " x ", size_val)
            fields["Size"] = size_val.strip(); continue
    if not fields["Type"]:
        for tok in parts:
            if ":" in tok or "-" in tok:
                continue
            if tok.strip():
                fields["Type"] = tok.strip(); break
    return fields

def extract_link_title_strict(page, rect, pad_px: float = 4.0, band_px: float = 28.0) -> str:
    import fitz
    if not rect: return ""
    r = fitz.Rect(rect).normalize()
    R = fitz.Rect(r.x0 - pad_px, r.y0 - pad_px, r.x1 + pad_px, r.y1 + pad_px)
    words = page.get_text("words") or []
    kept = []
    for x0, y0, x1, y1, w, *_ in words:
        if not w: continue
        cx = (x0 + x1)/2.0; cy = (y0 + y1)/2.0
        if R.contains(fitz.Point(cx, cy)):
            kept.append((y0, x0, w))
    if not kept:
        band = fitz.Rect(r.x0, r.y0 - band_px, r.x1, r.y1 + band_px)
        for x0, y0, x1, y1, w, *_ in words:
            if not w: continue
            cy_in = band.y0 <= ((y0 + y1)/2.0) <= band.y1
            x_overlaps = not (x1 < r.x0 or x0 > r.x1)
            if cy_in and x_overlaps:
                kept.append((y0, x0, w))
    kept.sort(key=lambda t:(round(t[0],3), t[1]))
    text = " ".join(t[2] for t in kept).strip()
    if not text:
        try: text = (page.get_textbox(r) or "").strip()
        except Exception: text = ""
    return _normalize_separators(text)

def split_position_and_title_start(raw: str) -> tuple[str,str]:
    s = (raw or "").strip()
    if not s: return "", ""
    m = POS_AT_START.match(s)
    if not m: return "", s
    pos = m.group(1)
    rest = s[m.end():].strip()
    m2 = NEXT_BULLET.search(rest)
    if m2: rest = rest[:m2.start()].strip()
    return pos, rest

ROOM_MAP_RAW = [
    ("Sink", "Plumbing"), ("Faucet", "Plumbing"), ("Sink Faucet", "Plumbing"), ("Shower", "Plumbing"),
    ("Shower Head", "Plumbing"), ("Tub", "Plumbing"), ("Shower System", "Plumbing"), ("Shower Drain", "Plumbing"),
    ("Pendant", "Lighting"), ("Sconce", "Lighting"), ("Sconces", "Lighting"), ("Lamp", "Lighting"),
]
ROOM_MAP = {k.lower(): v for (k, v) in ROOM_MAP_RAW}

def _infer_room_from_tag(tag_val: str) -> str:
    if not tag_val: return ""
    t = tag_val.strip().lower()
    if t in ROOM_MAP: return ROOM_MAP[t]
    best_key = ""
    for k in ROOM_MAP.keys():
        if t.startswith(k) and len(k) > len(best_key):
            best_key = k
    return ROOM_MAP.get(best_key, "")

def extract_links_by_pages(pdf_bytes: bytes, page_to_tag: dict[int,str]|None, only_listed_pages=True, pad_px:float=4.0, band_px:float=28.0)->pd.DataFrame:
    doc = fitz.open("pdf", pdf_bytes)
    rows=[]
    listed=set(page_to_tag.keys()) if page_to_tag else set()
    for pidx,page in enumerate(doc,start=1):
        if only_listed_pages and page_to_tag and pidx not in listed: continue
        tag_value=(page_to_tag or {}).get(pidx,"")
        for lnk in page.get_links():
            uri=(lnk.get("uri") or "").strip()
            if not uri.lower().startswith(("http://","https://")): continue
            rect=lnk.get("from")
            raw=extract_link_title_strict(page,rect,pad_px=pad_px,band_px=band_px)
            position,title=split_position_and_title_start(raw)
            fields=parse_link_title_fields(title)
            rows.append({"page":pidx,"Tags":tag_value,"Room":_infer_room_from_tag(tag_value),"Position":position,"Type":fields.get("Type",""),"QTY":fields.get("QTY",""),"Finish":fields.get("Finish",""),"Size":fields.get("Size",""),"link_url":uri,"link_text":title})
    return pd.DataFrame(rows)
