import os, io, re, json, time, requests
from typing import Optional, Tuple, Dict, Iterable, List, Any, Union, Set
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

# ----------------- Streamlit setup -----------------
st.set_page_config(page_title="Spec Link Extractor & Enricher", layout="wide")
st.title("🧰 Spec Link Extractor & Enricher")

# --- Persisted defaults to avoid rerun headaches ---
for k, v in {
    "pdf_bytes": None,
    "num_pages": None,
    "extracted_df": None,
    "pending_extract": False,  # gate extraction so Save doesn't re-extract
    # --- Skip/auto-skip controls for enrichment ---
    "skip_urls": [],                 # list[str] of URLs to skip
    "fail_counts": {},              # dict[url->int] failure counts
    "enable_auto_skip": True,       # auto add to skip after N fails
    "auto_skip_after_n": 2,         # N failures before auto-skip
}.items():
    st.session_state.setdefault(k, v)

# --- Progress timer helper ---
class Timer:
    def __enter__(self) -> 'Timer':
        self.t0: float = time.perf_counter()
        return self
    def __exit__(self, *exc: Any) -> None:
        self.dt: float = time.perf_counter() - self.t0

# ========================= Shared HTTP/parsing helpers =========================
PRICE_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
UA = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

# --- Safe scalar helper to avoid list/dict values leaking into DataFrame columns ---
def _first_scalar(v: Any) -> str:
    """Return a single string from possibly-list/dict values (e.g., meta['image'])."""
    if v is None:
        return ""
    # If list/tuple/set, pick the first non-empty str; else stringify first element
    if isinstance(v, (list, tuple, set)):
        for x in v:
            if isinstance(x, str) and x.strip():
                return x.strip()
        try:
            first = next(iter(v))
            return str(first)
        except StopIteration:
            return ""
    # If dict, try common keys
    if isinstance(v, dict):
        for k in ("url", "src", "image", "content", "@id"):
            vv = v.get(k)
            if isinstance(vv, str) and vv.strip():
                return vv.strip()
        return str(v)
    # If already a string or something else
    return v if isinstance(v, str) else str(v)

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

# --- Support spelled-out QTY values (e.g., QTY: TWO) ---
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
    # common shorthand
    "a": "1", "an": "1"
}

def _normalize_qty_token(token: str) -> str:
    if not token:
        return ""
    t = token.strip().lower().replace("-", " ")
    if t.isdigit():
        return t
    return NUMBER_WORDS.get(t, "")

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

def parse_link_title_fields(link_text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {"Type": "", "Quantity": "", "Finish/Color": "", "Dimensions": ""}
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
        if m and not fields["Quantity"]:
            q_raw = m.group(1).strip()
            q_norm = _normalize_qty_token(q_raw)
            if q_norm:
                fields["Quantity"] = q_norm
            else:
                q_up = q_raw.upper()
                fields["Quantity"] = "" if q_up == "XX" else q_up
            continue

        # Fallback: handle cases like "QTY: TWO" or "Quantity - THREE" when regex didn't catch digits
        if not fields["Quantity"] and ("QTY" in tok.upper() or "QUANTITY" in tok.upper()):
            after = tok
            if ":" in tok:
                after = tok.split(":", 1)[1]
            elif "-" in tok:
                after = tok.split("-", 1)[1]
            word = after.strip().split()[0] if after.strip() else ""
            q_norm = _normalize_qty_token(word)
            if q_norm:
                fields["Quantity"] = q_norm
                continue
        m = FINISH_RE.search(tok)
        if m and not fields["Finish/Color"]:
            fields["Finish/Color"] = m.group(1).strip(); continue
        m = SIZE_RE.search(tok)
        if m and not fields["Dimensions"]:
            size_val = m.group(1).strip()
            size_val = size_val.replace("”", '"').replace("“", '"').replace("’", "'").replace("‘", "'")
            size_val = re.sub(r"\s*[xX]\s*", " x ", size_val)
            fields["Dimensions"] = size_val.strip(); continue

    # If Type was unlabeled (e.g., "PENDANTS | QTY: 2 | ..."), infer from first unlabeled token
    if not fields["Type"]:
        for tok in parts:
            if ":" in tok or "-" in tok:  # likely a labeled field
                continue
            if tok.strip():
                fields["Type"] = tok.strip()
                break

    return fields

def extract_link_title_strict(page: fitz.Page, rect: fitz.Rect, pad_px: float = 4.0, band_px: float = 28.0) -> Tuple[str, Optional[fitz.Rect]]:
    """
    STRICT per-link capture but expanded to the *entire text line* that the link token sits on.
    Why: In Canva, a hyperlink can be applied to just the bullet (e.g., "2.") or a single
    token in a line. If we only keep words whose centers lie inside the rect, we can end up
    with rows like "2." or "5" instead of the full line. This version:
      1) Collect words whose center lies in the slightly padded rect (R).
      2) Expand to include *all words on the same (block,line)* as any kept word.
      3) As a final fallback, use page.get_textbox(rect).
    
    IMPORTANT: We require at least one word to be found within the link rectangle itself
    (not just nearby) to prevent associating links with unrelated adjacent text.
    Returns a tuple of (normalized_text, text_line_rect or None).
    """
    import fitz
    if not rect:
        return ""
    r = fitz.Rect(rect).normalize()
    R = fitz.Rect(r.x0 - pad_px, r.y0 - pad_px, r.x1 + pad_px, r.y1 + pad_px)

    # words: [x0,y0,x1,y1,word,block,line,word_no]
    words = page.get_text("words") or []

    # First, find words directly within the link rectangle (with padding)
    kept_in_rect = []  # Words actually in/very near the link rect
    kept = []  # (y0, x0, word, block, line) captured for line expansion
    for x0, y0, x1, y1, w, b, ln, *_ in words:
        if not w:
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if R.contains(fitz.Point(cx, cy)):
            kept_in_rect.append((y0, x0, w, b, ln))
            kept.append((y0, x0, w, b, ln))

    if not kept_in_rect:
        return "", None  # No text associated with this link - return empty to skip it

    text_rect = None
    if kept:
        # Expand to the full (block,line) of the kept word(s). Prefer the line with the most words.
        line_keys = [(b, ln) for *_ignore, b, ln in kept]
        counts = {}
        for key in line_keys:
            counts[key] = counts.get(key, 0) + 1
        best_key = max(counts.items(), key=lambda kv: kv[1])[0]
        bx, lx = best_key
        
        # Get all words on this line and calculate bounding box
        line_word_data = [(x0, y0, x1, y1, w) for x0, y0, x1, y1, w, b, ln, *_ in words if b == bx and ln == lx and w]
        
        if line_word_data:
            # Calculate the bounding box of the extracted text
            text_x_min = min(x0 for x0, y0, x1, y1, w in line_word_data)
            text_x_max = max(x1 for x0, y0, x1, y1, w in line_word_data)
            text_y_min = min(y0 for x0, y0, x1, y1, w in line_word_data)
            text_y_max = max(y1 for x0, y0, x1, y1, w in line_word_data)
            
            text_rect = fitz.Rect(text_x_min, text_y_min, text_x_max, text_y_max)

            # Check if link rectangle overlaps with text bounding box (with some tolerance)
            # If the link is too far from the text, it's probably not associated with it
            tolerance = 50  # pixels
            link_overlaps_text = not (r.x1 + tolerance < text_x_min or r.x0 - tolerance > text_x_max or 
                                      r.y1 + tolerance < text_y_min or r.y0 - tolerance > text_y_max)
            
            if not link_overlaps_text:
                # Link rectangle doesn't overlap with extracted text - probably wrong association
                return "", None
            
            # Sort and extract text
            line_words = [(y0, x0, w) for x0, y0, x1, y1, w in line_word_data]
            line_words.sort(key=lambda t: (round(t[0], 3), t[1]))
            text = " ".join(t[2] for t in line_words).strip()
        else:
            text = ""
    else:
        text = ""

    if not text:
        try:
            text = (page.get_textbox(R) or "").strip()
        except Exception:
            text = ""

    return _normalize_separators(text), text_rect

def split_position_and_title_start(raw: str) -> Tuple[str, str]:
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

ROOM_MAP: Dict[str, str] = {k.lower(): v for (k, v) in ROOM_MAP_RAW}
ROOM_OPTIONS = ["", "LIGHTING", "PLUMBING", "PAINT", "COUNTERTOPS + SLABS", "CABINETRY FINISHES", "HARDWARE", "TILE + STONE", "ACCENT MIRRORS", "DOORS, BASE, CASE", "WALLCOVERING", "APPLIANCES"]

# Type-to-Room mapping (hardcoded list from user)
TYPE_TO_ROOM_MAP_RAW = [
    # ACCENT MIRRORS
    ("MIRRORS", "ACCENT MIRRORS"),
    ("MIRROR", "ACCENT MIRRORS"),
    
    # COUNTERTOPS + SLABS
    ("COUNTERTOP", "COUNTERTOPS + SLABS"),
    ("PRO STORM QUARTZ", "COUNTERTOPS + SLABS"),
    ("COUNTERTOP + INSTALL", "COUNTERTOPS + SLABS"),
    ("SLAB", "COUNTERTOPS + SLABS"),
    ("BACKSPLASH", "COUNTERTOPS + SLABS"),
    ("COUNTERTOP INSTALL", "COUNTERTOPS + SLABS"),
    ("COUNTERTOP + FAB", "COUNTERTOPS + SLABS"),
    ("WATERFALL PANEL", "COUNTERTOPS + SLABS"),
    ("QUARTZITE", "COUNTERTOPS + SLABS"),
    
    # DOORS, BASE, CASE
    ("INTERIOR DOORS", "DOORS, BASE, CASE"),
    ("INTERIOR DOOR HINGES", "DOORS, BASE, CASE"),
    ("INTERIOR DOOR STOPS", "DOORS, BASE, CASE"),
    ("CASING", "DOORS, BASE, CASE"),
    ("INTERIOR DOOR HANDLES", "DOORS, BASE, CASE"),
    ("BASEBOARDS", "DOORS, BASE, CASE"),
    
    # HARDWARE
    ("HARDWARE", "HARDWARE"),
    ("APPLIANCE HARDWARE", "HARDWARE"),
    ("CABINETRY HARDWARE", "HARDWARE"),
    ("DOOR HARDWARE", "HARDWARE"),
    ("PULL", "HARDWARE"),
    ("KNOBS", "HARDWARE"),
    ("PULLS", "HARDWARE"),
    ("HANDLESET", "HARDWARE"),
    ("ENTRY SET", "HARDWARE"),
    ("TOWEL HOOKS", "HARDWARE"),
    
    # LIGHTING
    ("BEDROOM FANS", "LIGHTING"),
    ("CHANDELIER", "LIGHTING"),
    ("SCONCES", "LIGHTING"),
    ("PENDANT", "LIGHTING"),
    ("PENDANTS", "LIGHTING"),
    ("FAN", "LIGHTING"),
    ("FLUSH MOUNT", "LIGHTING"),
    ("SEMI-FLUSH", "LIGHTING"),
    ("WALL LAMP", "LIGHTING"),
    ("VANITY LIGHT", "LIGHTING"),
    ("ACCENT LIGHTING", "LIGHTING"),
    
    # PAINT
    ("INTERIOR PAINT", "PAINT"),
    ("LIMEWASH CEILING", "PAINT"),
    ("ALL OVER LIMEWASH", "PAINT"),
    ("TRIM PAINT", "PAINT"),
    ("WALL PAINT", "PAINT"),
    ("CEILING PAINT", "PAINT"),
    ("CABINETRY FINISHES", "PAINT"),
    
    # PLUMBING
    ("SINK FAUCET", "PLUMBING"),
    ("SINK FLANGE", "PLUMBING"),
    ("SINK", "PLUMBING"),
    ("SINK STRAINER", "PLUMBING"),
    ("POT FILLER", "PLUMBING"),
    ("TOILET", "PLUMBING"),
    ("BATHTUB", "PLUMBING"),
    ("TUB FILLER", "PLUMBING"),
    ("SHOWER SYSTEM", "PLUMBING"),
    ("HANDHELD SHOWER", "PLUMBING"),
    ("DRAIN", "PLUMBING"),
    ("TRIM KIT", "PLUMBING"),
    ("SPOUT", "PLUMBING"),
    ("VALVE TRIM", "PLUMBING"),
    ("SHOWER SPRAYERS", "PLUMBING"),
    ("SHOWER DRAIN", "PLUMBING"),
    ("TUB", "PLUMBING"),
    
    # TILE + STONE
    ("LVP", "TILE + STONE"),
    ("TUMBLED LIMESTONE", "TILE + STONE"),
    ("FLOORING", "TILE + STONE"),
    ("STONE WALL", "TILE + STONE"),
    ("MOSAIC", "TILE + STONE"),
    ("TILE", "TILE + STONE"),
    ("FLOOR TILE", "TILE + STONE"),
    ("WALL TILE", "TILE + STONE"),
    ("GROUT", "TILE + STONE"),
    ("THRESHOLD", "TILE + STONE"),
    ("TRIM PIECES", "TILE + STONE"),
    ("SHOWER PAN TILE", "TILE + STONE"),
    ("SHOWER WALLS + CURB TILE", "TILE + STONE"),
    ("SHOWER FLOOR TILE", "TILE + STONE"),
    ("ALL OVER FLOORING TILE", "TILE + STONE"),
    ("UPPER SHOWER TILE", "TILE + STONE"),
    ("CURB TILE", "TILE + STONE"),
    
    # WALLCOVERING
    ("WALLCOVERING", "WALLCOVERING"),
    ("WALLPAPER", "WALLCOVERING"),
    ("CEILING AND ALL OVER WALLPAPER", "WALLCOVERING"),
    ("VINYL WALLCOVERING", "WALLCOVERING"),
]

def _infer_room_from_type(type_val: str, type_to_room_map: Optional[Dict[str, str]] = None) -> str:
    """
    Given a 'Type' value (ex: "Pendant" / "Sink" / etc.), return the mapped Room.
    Uses the type_to_room_map if provided, otherwise uses default mapping.
    Strategy: exact lowercase match, then longest prefix match. Fallback: "" (blank)
    """
    if not type_val:
        return ""
    
    # Use provided map or default empty map
    if type_to_room_map is None:
        type_to_room_map = {}
    
    t = type_val.strip().lower()
    if t in type_to_room_map:
        return type_to_room_map[t]
    
    # Try longest prefix match
    best_key = ""
    for k in type_to_room_map.keys():
        if t.startswith(k) and len(k) > len(best_key):
            best_key = k
    return type_to_room_map.get(best_key, "")

def _infer_room_from_tag(tag_val: str) -> str:
    """
    Given a 'Tags' value from the page (ex: "Sink" / "Pendant" / etc.), return the mapped Room.
    Strategy: exact lowercase match, then longest prefix match. Fallback: "" (blank)
    """
    if not tag_val:
        return ""
    t = tag_val.strip().lower()
    if t in ROOM_MAP:
        return ROOM_MAP[t]
    best_key = ""
    for k in ROOM_MAP.keys():
        if t.startswith(k) and len(k) > len(best_key):
            best_key = k
    return ROOM_MAP.get(best_key, "")

from urllib.parse import urlparse as _urlparse_for_vendor

def _vendor_from_url(url: str) -> str:
    """
    Extract a simple vendor name from a URL.
    Example: https://www.wayfair.com/... -> 'wayfair'
    """
    if not url:
        return ""
    try:
        host = _urlparse_for_vendor(url).netloc.lower()
    except Exception:
        return ""
    # drop common prefixes like www, m, amp
    parts = [p for p in host.split(".") if p and p not in ("www", "m", "amp")]
    if not parts:
        return ""
    # simple heuristic: use the second-to-last part for typical .com domains
    if len(parts) >= 2:
        base = parts[-2]
    else:
        base = parts[0]
    return base


def extract_links_by_pages(
    pdf_bytes: bytes,
    page_to_tag: Optional[Dict[int, str]],
    page_to_room: Optional[Dict[int, str]] = None,
    only_listed_pages: bool = True,
    pad_px: float = 4.0,
    band_px: float = 28.0,
    view_mode: str = "trade",  # "trade" (old behavior) or "room"
    dedupe_by: str = "url_and_position",  # "url_and_position", "url_only", or "none"
    type_to_room_map: Optional[Dict[str, str]] = None,  # Map of type -> room for auto-filling
) -> Tuple[pd.DataFrame, int]:
    doc = fitz.open("pdf", pdf_bytes)
    rows = []
    # Track which text-line rectangles have already been consumed by a link so we
    # do not also add them later as "text block without link".
    used_text_rects = []
    listed = set(page_to_tag.keys()) if page_to_tag else set()
    
    # Deduplication tracking
    seen = set()  # Track (page, canonical_url, position) or (page, canonical_url) depending on dedupe_by
    duplicates_skipped = 0  # Count of duplicates removed

    for pidx, page in enumerate(doc, start=1):
        if only_listed_pages and page_to_tag and pidx not in listed:
            continue

        tag_value = (page_to_tag or {}).get(pidx, "")
        room_value = (page_to_room or {}).get(pidx, _infer_room_from_tag(tag_value))

        # Get page dimensions to exclude top-right region (where titles are)
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        top_right_x_min = page_width * 0.6  # Right 40%
        top_right_y_max = page_height * 0.25  # Top 25%

        # Build a map of link rectangles to URLs for this page
        # This allows us to check if a text block has an associated link
        link_rects = []  # List of (rect, uri) tuples
        
        # Process all links (existing behavior - items with links)
        processed_rects = []  # Track which link rects we've processed (as Rect objects)
        
        for lnk in page.get_links():
            uri = (lnk.get("uri") or "").strip()
            if not uri.lower().startswith(("http://", "https://")):
                continue

            # Canonicalize URL for deduplication
            canonical_uri = canonicalize_url(uri)

            rect = lnk.get("from")
            if not rect:
                continue
                
            r = fitz.Rect(rect).normalize()
            
            # Skip links in top-right region (where page titles are)
            link_center_x = (r.x0 + r.x1) / 2.0
            link_center_y = (r.y0 + r.y1) / 2.0
            if link_center_x >= top_right_x_min and link_center_y <= top_right_y_max:
                continue  # Skip top-right region
            
            processed_rects.append(r)
            link_rects.append((r, uri))
            
            raw, text_rect = extract_link_title_strict(page, rect, pad_px=pad_px, band_px=band_px)
            position, title = split_position_and_title_start(raw)

            # Require that at least some text is actually within the link rectangle
            # This prevents associating links with nearby unrelated text
            if not raw:
                # No text found for this link - skip it to avoid false associations
                continue

            # Ignore common headings like "MATERIALS LIST"
            if not title or title.strip().lower().startswith(("materials list", "material list")):
                continue

            # Deduplication check
            if dedupe_by == "url_and_position":
                # If we failed to read a position, fall back to URL-only dedupe on this page
                dedupe_key = (pidx, canonical_uri, position) if position else (pidx, canonical_uri)
            elif dedupe_by == "url_only":
                dedupe_key = (pidx, canonical_uri)
            else:  # "none"
                dedupe_key = None
            
            if dedupe_key and dedupe_key in seen:
                duplicates_skipped += 1
                continue  # Skip duplicate
            
            if dedupe_key:
                seen.add(dedupe_key)
            
            if text_rect:
                # Mark this text line as consumed by a link to prevent double-capturing
                used_text_rects.append(text_rect)

            fields = parse_link_title_fields(title)
            type_col = fields.get("Type", "")

            # --- View-mode mapping logic ---
            # Trade View (existing behavior):
            #   Room  = room_value (inferred / dropdown), but auto-fill from Type if available
            #   Type  = parsed from link text (fields["Type"])
            #
            # Room View (new behavior):
            #   Room  = tag_value (what you entered in the Tags table)
            #   Type  = parsed from link text initially; you will overwrite via dropdown in UI
            if view_mode == "room":
                room_col = tag_value
            else:
                # Try to auto-fill Room from Type first, then fall back to room_value
                room_from_type = _infer_room_from_type(type_col, type_to_room_map)
                room_col = room_from_type if room_from_type else room_value

            rows.append({
                "page": pidx,
                "Tags": tag_value,
                "Room": room_col,
                "Position": position,
                "Type": type_col,
                "Quantity": fields.get("Quantity", ""),
                "Finish/Color": fields.get("Finish/Color", ""),
                "Dimensions": fields.get("Dimensions", ""),
                "Product Website": uri,
                "link_text": title,
                "Client Product Name": f"{tag_value.strip()} {fields.get('Type', '').strip()}".strip(),
                "Vendor": _vendor_from_url(uri),
            })
        
        # Now process text blocks that don't have links
        # Get text blocks and check if they overlap with any link rectangle
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] != 0:  # block type 0 is text, others are images
                continue
            
            block_rect = fitz.Rect(block[0:4])  # x0, y0, x1, y1
            block_text = block[4].strip()  # text content
            
            # Skip text in top-right region (where page titles are)
            block_center_x = (block_rect.x0 + block_rect.x1) / 2.0
            block_center_y = (block_rect.y0 + block_rect.y1) / 2.0
            if block_center_x >= top_right_x_min and block_center_y <= top_right_y_max:
                continue  # Skip top-right region
            
            # Skip if this block overlaps with a link we already processed
            overlaps_link = False
            for link_rect in processed_rects:
                try:
                    if block_rect.intersects(link_rect):
                        overlaps_link = True
                        break
                except Exception:
                    # If intersection check fails, assume no overlap
                    pass
            overlaps_used_text = False
            for tr in used_text_rects:
                try:
                    # light padding to catch near-identical lines
                    expanded_tr = fitz.Rect(tr.x0 - 2, tr.y0 - 2, tr.x1 + 2, tr.y1 + 2)
                    if block_rect.intersects(expanded_tr):
                        overlaps_used_text = True
                        break
                except Exception:
                    pass
            
            if overlaps_link or overlaps_used_text or not block_text:
                continue
            
            # Skip common headings
            if block_text.lower().startswith(("materials list", "material list")):
                continue
            
            # This is a text block without a link - extract it with blank URL
            position, title = split_position_and_title_start(block_text)
            
            # Skip if no meaningful title
            if not title:
                continue
            
            fields = parse_link_title_fields(title)
            type_col = fields.get("Type", "")
            
            if view_mode == "room":
                room_col = tag_value
            else:
                # Try to auto-fill Room from Type first, then fall back to room_value
                room_from_type = _infer_room_from_type(type_col, type_to_room_map)
                room_col = room_from_type if room_from_type else room_value
            
            rows.append({
                "page": pidx,
                "Tags": tag_value,
                "Room": room_col,
                "Position": position,
                "Type": type_col,
                "Quantity": fields.get("Quantity", ""),
                "Finish/Color": fields.get("Finish/Color", ""),
                "Dimensions": fields.get("Dimensions", ""),
                "Product Website": "",  # Blank since no link
                "link_text": title,
                "Client Product Name": f"{tag_value.strip()} {fields.get('Type', '').strip()}".strip(),
                "Vendor": "",
            })

    return pd.DataFrame(rows), duplicates_skipped


# ========================= Tabs 2/3: Your Firecrawl + parsers =========================
# --- Title normalization helper (keeps just the product name like 'Sawyer Chandelier') ---
from urllib.parse import urlparse as _urlparse_for_title

def normalize_product_title(raw: str, url: Optional[str] = None) -> str:
    if not raw:
        return ""
    t = str(raw).strip()
    # derive site token from URL (e.g., 'lumens')
    site = ""
    if url:
        try:
            host = _urlparse_for_title(url).netloc.lower()
            parts = [p for p in host.split(".") if p not in ("www", "m", "amp")]
            if parts:
                site = parts[-2] if len(parts) >= 2 else parts[0]
        except Exception:
            site = ""
    # drop trailing site mentions like 'at Lumens.com' / '| Lumens'
    tl = t.lower()
    if site:
        for sep in [" at ", " | ", " - ", " — ", " – "]:
            needle = sep + site
            pos = tl.find(needle)
            if pos != -1:
                t = t[:pos].strip(); tl = t.lower(); break
        if tl.endswith(site):
            t = t[: -len(site)].strip(); tl = t.lower()
    # keep first chunk before separators
    for sep in [" | ", " - ", " — ", " – "]:
        if sep in t:
            t = t.split(sep)[0].strip(); break
    # remove trailing 'by BRAND'
    low = t.lower()
    if " by " in low:
        t = t[: low.find(" by ")].strip()
    # normalize whitespace/quotes
    t = " ".join(t.split()).strip('\"\'')
    return t

def extract_title_from_html(meta: Dict[str, Any], html: str) -> str:
    """Best-effort product/page title from meta + JSON-LD + visible H1/Title."""
    title = ""
    for k in ("og:title", "twitter:title", "title"):
        v = (meta or {}).get(k)
        if isinstance(v, (list, tuple)):
            v = _first_scalar(v)
        if isinstance(v, str) and v.strip():
            title = v.strip(); break
    if not title and html:
        try:
            soup = BeautifulSoup(html or "", "lxml")
            for tag in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(tag.string or "")
                except Exception:
                    continue
                objs = data if isinstance(data, list) else [data]
                for o in objs:
                    t = o.get("@type")
                    if t == "Product" or (isinstance(t, list) and "Product" in t):
                        nm = o.get("name")
                        if isinstance(nm, str) and nm.strip():
                            return nm.strip()
            h1 = soup.find("h1")
            if h1 and h1.get_text(strip=True):
                return h1.get_text(strip=True)
            if soup.title and soup.title.string:
                return soup.title.string.strip()
        except Exception:
            pass
    return title.strip()

def pick_image_and_price_bs4(html: str, base_url: str) -> Tuple[str, str, str]:
    """Lightweight fallback: og/twitter → JSON-LD → meta → visible $ pattern + title."""
    soup = BeautifulSoup(html or "", "lxml")
    meta_map = {}
    for m in soup.find_all("meta"):
        name = (m.get("property") or m.get("name") or "").strip()
        content = m.get("content")
        if name and content:
            meta_map[name] = content

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
        # Use broader HTML scan (srcset + lazyload)
        img_url = _first_image_from_html(html, base_url)

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

    title = extract_title_from_html(meta_map, str(soup))
    return img_url or "", price or "", title or ""

# --------- Lumens-specific helpers (targets PDP-large + lazyload) ----------
LUMENS_PDP_RE = re.compile(
    r'https://images\.lumens\.com/is/image/Lumens/[A-Za-z0-9_/-]+?\?[^\s]*PDP-(?:large|small)[^\s]*',
    re.I
)

def _upgrade_lumens_image_url(u: str) -> str:
    """Convert PDP-small (or sized variants) to PDP-large when possible."""
    if not isinstance(u, str):
        return ""
    u = u.strip()
    if not u:
        return ""
    if "PDP-large" in u:
        return u
    if "PDP-small" in u:
        return u.replace("PDP-small", "PDP-large")
    return u

def _clean_image_url(u: str) -> str:
    """Strip common trailing junk like a closing parenthesis or quotes."""
    if not isinstance(u, str):
        return ""
    u = u.strip().strip(')"\'')
    return u

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

def _first_image_from_html(html: str, base_url: str = "") -> str:
    """
    Best-effort image extraction from HTML:
    - prefer <picture><source srcset> largest width
    - then <img> with srcset/data-srcset (largest)
    - then common lazyload attrs (data-zoom-image, data-large_image, data-original, data-src, data-lazy)
    - then plain src
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    candidates = []

    def _add(url_val: str):
        if isinstance(url_val, str) and url_val.strip():
            candidates.append(urljoin(base_url, url_val.strip()))

    def _pick_srcset(val: str):
        cand = _largest_from_srcset(val)
        if cand:
            _add(cand)

    # <picture> sources first
    for pict in soup.find_all("picture"):
        for src in pict.find_all("source"):
            for attr in ("srcset", "data-srcset"):
                v = src.get(attr)
                if isinstance(v, str) and v.strip():
                    _pick_srcset(v)
    # <img> tags: srcset then lazy attrs
    for im in soup.find_all("img"):
        for attr in ("srcset", "data-srcset"):
            v = im.get(attr)
            if isinstance(v, str) and v.strip():
                _pick_srcset(v)
        for attr in ("data-zoom-image", "data-large_image", "data-original", "data-src", "data-lazy", "src"):
            v = im.get(attr)
            if isinstance(v, str) and v.strip():
                _add(v)
    # preload hints
    preload = soup.find("link", rel=lambda v: v and "preload" in v, attrs={"as": "image"})
    if preload and isinstance(preload.get("href"), str):
        _add(preload["href"])

    return candidates[0] if candidates else ""

def _first_lumens_pdp_large_from_html(html: str) -> str:
    if not html: return ""
    m = LUMENS_PDP_RE.search(html)
    if m:
        return _upgrade_lumens_image_url(m.group(0))

    soup = BeautifulSoup(html, "lxml")
    for im in soup.find_all("img"):
        for attr in ("data-src", "data-original", "data-zoom-image", "data-large_image", "src"):
            v = im.get(attr)
            if isinstance(v, str) and "$Lumens.com-PDP-large$" in v:
                return v
            if isinstance(v, str) and "$Lumens.com-PDP-small$" in v:
                return _upgrade_lumens_image_url(v)
        ss = im.get("srcset") or im.get("data-srcset")
        if isinstance(ss, str) and "$Lumens.com-PDP-large$" in ss:
            cand = _largest_from_srcset(ss)
            if cand: return cand
        if isinstance(ss, str) and "$Lumens.com-PDP-small$" in ss:
            cand = _largest_from_srcset(ss)
            if cand: return _upgrade_lumens_image_url(cand)

    for pict in soup.find_all("picture"):
        for src in pict.find_all("source"):
            ss = src.get("srcset") or src.get("data-srcset")
            if isinstance(ss, str) and "$Lumens.com-PDP-large$" in ss:
                cand = _largest_from_srcset(ss)
                if cand: return cand
            if isinstance(ss, str) and "$Lumens.com-PDP-small$" in ss:
                cand = _largest_from_srcset(ss)
                if cand: return _upgrade_lumens_image_url(cand)

    preload = soup.find("link", rel=lambda v: v and "preload" in v, attrs={"as": "image"})
    if preload and isinstance(preload.get("href"), str) and "$Lumens.com-PDP-large$" in preload["href"]:
        return preload["href"]
    if preload and isinstance(preload.get("href"), str) and "$Lumens.com-PDP-small$" in preload["href"]:
        return _upgrade_lumens_image_url(preload["href"])

    return ""

def parse_image_and_price_lumens_from_v2(scrape: Dict[str, Any], base_url: str = "") -> Tuple[str, str, str]:
    """
    Lumens-specific parser: prioritizes structured JSON, then Lumens PDP-large images.
    Returns (img, price, title).
    """
    if not scrape:
        return "", "", ""

    data = scrape.get("data") or {}
    html = data.get("html") or ""
    md = data.get("markdown") or ""

    # Priority 1: Use structured JSON extraction from Firecrawl
    json_data = data.get("json") or {}
    if isinstance(json_data, dict):
        content = json_data.get("content") if isinstance(json_data.get("content"), dict) else json_data

        img = _first_scalar(content.get("imageUrl", ""))
        price = _first_scalar(content.get("price", ""))
        title = _first_scalar(content.get("productName", ""))

        # For Lumens, upgrade image to PDP-large if we got a smaller version
        if img and "lumens.com" in img.lower():
            img = _upgrade_lumens_image_url(img)

        # Ensure price has currency symbol
        if price and not price.startswith("$"):
            price = f"${price}"

        # If we got all data from JSON, return it
        if img and price and title:
            return _clean_image_url(img), price, normalize_product_title(title, base_url)
    else:
        img = price = title = ""

    # Priority 2: Lumens-specific PDP-large image search in markdown/HTML
    if not img:
        if isinstance(md, str):
            m = LUMENS_PDP_RE.search(md)
            if m:
                img = _clean_image_url(_upgrade_lumens_image_url(m.group(0)))

        if not img:
            img = _clean_image_url(_first_lumens_pdp_large_from_html(html))

        # Fallback to generic HTML image scan
        if not img:
            img = _clean_image_url(_first_image_from_html(html, base_url))

    # Priority 3: Price from HTML parsing (only if not in JSON)
    if not price and html:
        soup = BeautifulSoup(html, "lxml")

        # JSON-LD structured data
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                obj = json.loads(tag.string or "")
                objs = obj if isinstance(obj, list) else [obj]
                for o in objs:
                    t = o.get("@type")
                    if t == "Product" or (isinstance(t, list) and "Product" in t):
                        offers = o.get("offers") or {}
                        if isinstance(offers, list):
                            offers = offers[0] if offers else {}
                        p = offers.get("price") or (offers.get("priceSpecification") or {}).get("price")
                        if p:
                            price = p if str(p).startswith("$") else f"${p}"
                            break
                if price:
                    break
            except Exception:
                continue

        # Meta tag price
        if not price:
            m = soup.find("meta", attrs={"itemprop": "price"}) or \
                soup.find("meta", attrs={"property": "product:price:amount"})
            if m and m.get("content"):
                val = m["content"]
                price = val if str(val).startswith("$") else f"${val}"

        # Regex price search
        if not price:
            m = PRICE_RE.search(soup.get_text(" ", strip=True))
            if m:
                price = m.group(0)

    # Title extraction
    if not title:
        title = extract_title_from_html(data.get("metadata") or {}, html)

    return img or "", price or "", title or ""

# ----------------- Firecrawl v2 (REST) helpers -----------------
def firecrawl_scrape_v2(url: str, api_key: str, mode: str = "simple") -> Dict[str, Any]:
    """
    Call Firecrawl /v2/scrape via REST with comprehensive product data extraction.
    Uses a structured schema to extract all product details in one call.
    """
    if not api_key:
        return {}

    # Comprehensive schema for product data extraction
    product_schema = {
        "type": "object",
        "properties": {
            "productName": {
                "type": "string",
                "description": "Full product name or title, without site name or extra text"
            },
            "price": {
                "type": "string",
                "description": "Product price with currency symbol (e.g., $299.00)"
            },
            "brand": {
                "type": "string",
                "description": "Brand or manufacturer name"
            },
            "imageUrl": {
                "type": "string",
                "description": "Main product image URL - prefer largest/highest quality version"
            },
            "description": {
                "type": "string",
                "description": "Product description or summary"
            },
            "dimensions": {
                "type": "string",
                "description": "Product dimensions or size (e.g., 24 x 12 x 8 inches)"
            },
            "finish": {
                "type": "string",
                "description": "Product finish, color, or material"
            },
            "availability": {
                "type": "string",
                "description": "Stock status (in stock, out of stock, etc.)"
            }
        },
        "required": []
    }

    payload = {
        "url": url,
        "formats": [
            "html",
            "markdown",
            {"type": "json", "schema": product_schema}
        ],
        "proxy": "auto",
        "timeout": 45000,
    }

    # Simplified mode system - just standard and deep
    if mode == "deep":
        payload["actions"] = [
            {"type": "wait", "milliseconds": 1500},
            {"type": "scroll", "y": 2000},
            {"type": "wait", "milliseconds": 2000},
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

def parse_image_and_price_from_v2_generic(scrape: Dict[str, Any], base_url: str = "") -> Tuple[str, str, str]:
    """
    Parse product data from Firecrawl v2 response.
    Prioritizes structured JSON extraction, falls back to metadata and HTML parsing.
    Returns (image_url, price, title).
    """
    if not scrape:
        return "", "", ""

    data = scrape.get("data") or {}

    # Priority 1: Use structured JSON extraction from Firecrawl
    json_data = data.get("json") or {}
    if isinstance(json_data, dict):
        # Handle nested content structure that some Firecrawl responses have
        content = json_data.get("content") if isinstance(json_data.get("content"), dict) else json_data

        img = _first_scalar(content.get("imageUrl", ""))
        price = _first_scalar(content.get("price", ""))
        title = _first_scalar(content.get("productName", ""))

        # Ensure price has currency symbol
        if price and not price.startswith("$"):
            price = f"${price}"

        # If we got all data from JSON, return it
        if img and price and title:
            return img, price, normalize_product_title(title, base_url)
    else:
        img = price = title = ""

    # Priority 2: Metadata (og:image, twitter, etc.)
    meta = data.get("metadata") or {}
    html = data.get("html") or ""

    if not img:
        img = _first_scalar(meta.get("og:image") or meta.get("twitter:image") or meta.get("image") or "")

    if not title:
        title = extract_title_from_html(meta, html)

    # Priority 3: HTML parsing fallback (only if still missing data)
    if (not img or not price) and html:
        soup = BeautifulSoup(html, "lxml")

        # Image from HTML
        if not img:
            img = _first_image_from_html(html, base_url)

        # Price from JSON-LD or visible text
        if not price:
            for tag in soup.find_all("script", type="application/ld+json"):
                try:
                    obj = json.loads(tag.string or "")
                    objs = obj if isinstance(obj, list) else [obj]
                    for o in objs:
                        t = o.get("@type")
                        if t == "Product" or (isinstance(t, list) and "Product" in t):
                            offers = o.get("offers") or {}
                            if isinstance(offers, list):
                                offers = offers[0] if offers else {}
                            p = offers.get("price") or (offers.get("priceSpecification") or {}).get("price")
                            if p:
                                price = p if str(p).startswith("$") else f"${p}"
                                break
                    if price:
                        break
                except Exception:
                    continue

            # Last resort: regex price search
            if not price:
                m = PRICE_RE.search(soup.get_text(" ", strip=True))
                if m:
                    price = m.group(0)

    return _first_scalar(img) or "", _first_scalar(price) or "", _first_scalar(title) or ""

def enrich_domain_firecrawl_v2(url: str, api_key: str) -> Tuple[str, str, str, str]:
    """
    Generic domain enrichment using Firecrawl v2.
    Tries standard scrape first, then deep scrape if needed.
    """
    # Try standard scrape first
    sc = firecrawl_scrape_v2(url, api_key, mode="simple")
    img, price, title = parse_image_and_price_from_v2_generic(sc, url)
    title = normalize_product_title(title, url)
    status = "firecrawl_v2"

    # If missing any data, try deep scrape with scrolling
    if (not img or not price or not title) and api_key:
        sc2 = firecrawl_scrape_v2(url, api_key, mode="deep")
        i2, p2, t2 = parse_image_and_price_from_v2_generic(sc2, url)
        img = img or i2
        price = price or p2
        title = title or normalize_product_title(t2, url)
        if i2 or p2 or t2:
            status = "firecrawl_v2_deep"

    return img, price, title, status

def enrich_wayfair_v2(url: str, api_key: str) -> Tuple[str, str, str, str]:
    return enrich_domain_firecrawl_v2(url, api_key)

def enrich_ferguson_v2(url: str, api_key: str) -> Tuple[str, str, str, str]:
    return enrich_domain_firecrawl_v2(url, api_key)

def enrich_lumens_v2(url: str, api_key: str) -> Tuple[str, str, str, str]:
    """
    Lumens-specific enrichment with PDP-large image optimization.
    Uses structured JSON extraction first, then Lumens-specific fallbacks.
    """
    u = canonicalize_url(url)

    # Try standard scrape first
    sc = firecrawl_scrape_v2(u, api_key, mode="simple")
    img, price, title = parse_image_and_price_lumens_from_v2(sc, u)
    img = _clean_image_url(img)
    title = normalize_product_title(title, u)
    status = "firecrawl_v2"

    # If missing any data, try deep scrape
    if (not img or not price or not title) and api_key:
        sc2 = firecrawl_scrape_v2(u, api_key, mode="deep")
        i2, p2, t2 = parse_image_and_price_lumens_from_v2(sc2, u)
        img = img or _clean_image_url(i2)
        price = price or p2
        title = title or normalize_product_title(t2, u)
        if i2 or p2 or t2:
            status = "firecrawl_v2_deep"

    # Final fallback: direct HTTP request with BeautifulSoup (rare edge cases)
    if not img or not title:
        r = requests_get(u)
        if r and r.text:
            i3, p3, t3 = pick_image_and_price_bs4(r.text, u)
            img = img or _clean_image_url(_first_scalar(i3))
            price = price or _first_scalar(p3)
            title = title or normalize_product_title(_first_scalar(t3), u)
            if i3 or p3 or t3:
                status = status + "+bs4" if status else "bs4"

    return img, price, title, status

# ----------------- Bulk Enrichment (chunked + resume) -----------------

def enrich_urls(df: pd.DataFrame, url_col: str, api_key: Optional[str], *, max_per_run: int = 100, start_at: int = 0, autosave_every: int = 25) -> pd.DataFrame:
    """Enrich a links CSV in chunks with resume + autosave.
    Ensures all output columns are homogenous strings (no lists/dicts) to avoid Arrow errors.
    """
    out = df.copy()

    # Pick URL column safely
    if url_col not in out.columns:
        if len(out.columns) >= 2:
            url_col = out.columns[1]
        else:
            st.error(f"URL column '{url_col}' not found.")
            return out

    # Ensure output columns exist and are string-typed.
    # Backward compatibility: if older 'product_title' column exists, reuse it as 'Product Name'.
    if "Product Name" not in out.columns and "product_title" in out.columns:
        out["Product Name"] = out["product_title"]
    for col in ("scraped_image_url", "price", "scrape_status", "Product Name"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str).fillna("")

    urls = out[url_col].astype(str).fillna("").tolist()
    imgs   = out["scraped_image_url"].astype(str).fillna("").tolist()
    prices = out["price"].astype(str).fillna("").tolist()
    status = out["scrape_status"].astype(str).fillna("").tolist()
    titles = out["Product Name"].astype(str).fillna("").tolist()

    # Done = has image AND price AND title
    done = [bool(imgs[i]) and bool(prices[i]) and bool(titles[i]) for i in range(len(urls))]
    pending_idxs = [i for i, ok in enumerate(done) if not ok]

    if not pending_idxs:
        st.info("Nothing to do — all rows already enriched.")
        return out

    # Apply resume window
    start_at = max(0, int(start_at))
    max_per_run = max(1, int(max_per_run))
    window = pending_idxs[start_at : start_at + max_per_run]
    if not window:
        st.info("No pending rows in the selected window (adjust 'Skip first N pending').")
        return out

    api_key = (api_key or "").strip()
    prog = st.progress(0)
    hb = st.empty()

    t0 = time.perf_counter()
    for k, i in enumerate(window, start=1):
        # --- Skip list + auto-skip settings ---
        skip_set = set(st.session_state.get("skip_urls", []))
        enable_auto = bool(st.session_state.get("enable_auto_skip", True))
        auto_n = int(st.session_state.get("auto_skip_after_n", 2))
        u = urls[i].strip()
        if not u:
            imgs[i] = ""; prices[i] = ""; titles[i] = titles[i] or "NAME NEEDED"; status[i] = (status[i] + "+no_url+name_needed") if status[i] else "no_url+name_needed"
            prog.progress(k/len(window));
            continue
        # If this URL is manually skipped, mark and move on
        if u in skip_set:
            titles[i] = titles[i] or "NAME NEEDED"
            status[i] = (status[i] + "+skipped+name_needed") if status[i] else "skipped+name_needed"
            prog.progress(k/len(window));
            continue

        img = price = title = ""; st_code = ""
        try:
            if api_key:
                if "lumens.com" in u:
                    img, price, title, st_code = enrich_lumens_v2(u, api_key)
                elif "fergusonhome.com" in u:
                    img, price, title, st_code = enrich_ferguson_v2(u, api_key)
                elif "wayfair.com" in u:
                    img, price, title, st_code = enrich_wayfair_v2(u, api_key)
                else:
                    img, price, title, st_code = enrich_domain_firecrawl_v2(u, api_key)

            if not img or not price or not title:
                r = requests_get(u)
                if r and r.text:
                    i2, p2, t2 = pick_image_and_price_bs4(r.text, u)
                    img = img or _first_scalar(i2)
                    price = price or _first_scalar(p2)
                    title = title or normalize_product_title(_first_scalar(t2), u)
                    st_code = (st_code + "+bs4_ok") if st_code else "bs4_ok"
                else:
                    st_code = (st_code + "+fetch_failed") if st_code else "fetch_failed"
        except Exception as e:
            st_code = (st_code + "+error") if st_code else "error"

        # If we still don't have a title, mark it explicitly so it's easy to fix later
        if not title:
            title = "NAME NEEDED"
            st_code = (st_code + "+name_needed") if st_code else "name_needed"

        # --- Track failures and auto-skip if needed on future passes ---
        if ("fetch_failed" in st_code) or ("error" in st_code):
            fc = st.session_state.get("fail_counts", {})
            fc[u] = int(fc.get(u, 0)) + 1
            st.session_state["fail_counts"] = fc
            if enable_auto and fc[u] >= auto_n and u not in st.session_state.get("skip_urls", []):
                st.session_state["skip_urls"].append(u)
                st.toast(f"Auto-skipped after {fc[u]} failures: {u}", icon="⏭️")

        # Force scalars (strings)
        imgs[i] = _first_scalar(img) or ""
        prices[i] = _first_scalar(price) or ""
        titles[i] = _first_scalar(title) or ""
        status[i] = _first_scalar(st_code) or ""

        # Heartbeat + ETA
        elapsed = time.perf_counter() - t0
        rate = elapsed / max(1, k)
        remaining = (len(window) - k) * rate
        hb.write(f"Processed {k}/{len(window)} • last URL: {u[:80]} • ETA ~{int(remaining)}s")
        prog.progress(k/len(window))

        # Autosave
        if autosave_every and (k % autosave_every == 0):
            tmp = out.copy()
            tmp["scraped_image_url"] = imgs
            tmp["price"] = prices
            tmp["scrape_status"] = status
            tmp["Product Name"] = titles
            st.session_state["last_partial_csv"] = tmp.to_csv(index=False).encode("utf-8")
            st.toast("Autosaved partial CSV", icon="💾")

        # gentle spacing for UI
        time.sleep(0.05)

    # Write back arrays to the DataFrame (strings only)
    out["scraped_image_url"], out["price"], out["scrape_status"], out["Product Name"] = imgs, prices, status, titles

    # Persist updated skip/failure state
    st.session_state["skip_urls"] = st.session_state.get("skip_urls", [])
    st.session_state["fail_counts"] = st.session_state.get("fail_counts", {})
    return out

# ----------------- Get API key from environment only -----------------
# API key is read from Streamlit secrets (set in Cloud → Settings → Secrets)
# Add this line to your secrets.toml: FIRECRAWL_API_KEY = "your-key-here"
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

# ----------------- Tabs -----------------
tab1, tab2, tab3 = st.tabs([
    "1) Extract from PDF (pages + Tags + Position + full titles)",
    "2) Enrich CSV (Image URL + Price)",
    "3) Test single URL"
])

# Helper for the Extract button
def _start_extract() -> None:
    st.session_state["pending_extract"] = True

# --- Tab 1: Canva PDF → rows ---
with tab1:
    st.caption("Build a page→Tags table, then extract ALL web links on those pages. Each link stays on its own row.")
    pdf_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_extractor")

    # Cache PDF bytes + page count as soon as the user uploads
    if pdf_file:
        try:
            _peek = fitz.open("pdf", pdf_file.getvalue())
            st.session_state["pdf_bytes"] = pdf_file.getvalue()
            st.session_state["num_pages"] = len(_peek)
            st.info(f"PDF detected with **{st.session_state['num_pages']}** page(s).")
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

    dedupe_mode = st.selectbox(
        "Deduplication mode",
        options=["url_and_position", "url_only", "none"],
        index=0,
        format_func=lambda x: {
            "url_and_position": "Remove duplicates: same URL + same position on same page (recommended)",
            "url_only": "Remove duplicates: same URL on same page (keeps different positions)",
            "none": "No deduplication (keep all links)"
        }[x],
        help="Choose how to handle duplicate links. 'url_and_position' is recommended to remove true duplicates while preserving legitimate multiple occurrences.",
        key="dedupe_mode_select"
    )
    # Store in session state for use in extraction
    st.session_state["dedupe_mode"] = dedupe_mode

    # Initialize type-to-room mapping in session state (hardcoded list)
    if "type_to_room_map" not in st.session_state:
        st.session_state["type_to_room_map"] = {k.lower(): v for k, v in TYPE_TO_ROOM_MAP_RAW}

    st.button(
        "Extract",
        type="primary",
        key="extract_btn",
        disabled=(st.session_state.get("pdf_bytes") is None),
        on_click=_start_extract,
    )

    # Perform extraction after the UI is declared so button clicks set the flag first
    if st.session_state.get("pending_extract") and st.session_state.get("pdf_bytes") is not None:
        # Build mapping from editor
        page_to_tag: Dict[int, str] = {}
        num_pages = st.session_state.get("num_pages")
        try:
            # mapping_df is defined in the same script run; guard if not
            for _, row in mapping_df.iterrows():
                p_raw = str(row.get("page", "")).strip()
                t_raw = str(row.get("Tags", "")).strip()
                if not p_raw:
                    continue
                try:
                    p_no = int(p_raw)
                    if p_no >= 1 and (num_pages is None or p_no <= num_pages):
                        page_to_tag[p_no] = t_raw
                except Exception:
                    continue
        except NameError:
            # If mapping_df isn't in scope (rare), just extract all pages without tags
            page_to_tag = {}

        with st.spinner("Extracting links, positions & titles…"):
            # Get dedupe_mode from session state (set by selectbox above)
            dedupe_mode_value = st.session_state.get("dedupe_mode", "url_and_position")
            # Get type-to-room mapping from session state
            type_to_room_map_value = st.session_state.get("type_to_room_map", {})
            
            df, duplicates_skipped = extract_links_by_pages(
                st.session_state["pdf_bytes"], page_to_tag, None,
                only_listed_pages=only_listed,
                pad_px=4.0,
                band_px=28.0,
                dedupe_by=dedupe_mode_value,
                type_to_room_map=type_to_room_map_value
            )
        st.session_state["extracted_df"] = df if not df.empty else None
        st.session_state["pending_extract"] = False
        
        # Show duplicate removal feedback
        if duplicates_skipped > 0:
            st.info(f"✅ Removed {duplicates_skipped} duplicate link(s) during extraction. Extracted {len(df)} unique link(s).")
        elif dedupe_mode_value != "none":
            st.success(f"✅ Extracted {len(df)} unique link(s).")

    # Always render editable table if we have data (only in Tab 1)
    if st.session_state.get("extracted_df") is not None:
        st.caption("Edit the Room per row if needed, then click **Save room edits**. When you're done, download the CSV.")

        df_show = st.session_state["extracted_df"].copy()
        if "Room" not in df_show.columns:
            df_show["Room"] = ""
        df_show["Room"] = df_show["Room"].astype(str).fillna("").replace({"nan": ""})

        col_cfg = {
            "Room": st.column_config.SelectboxColumn(
                "Room",
                options=ROOM_OPTIONS,
                help="Choose a room/category or leave blank",
            ),
            "page": st.column_config.TextColumn("page", disabled=True),
            "Tags": st.column_config.TextColumn("Tags", disabled=True),
            "Position": st.column_config.TextColumn("Position", disabled=True),
            "Type": st.column_config.TextColumn("Type", disabled=True),
            "Quantity": st.column_config.TextColumn("Quantity", disabled=True),
            "Finish/Color": st.column_config.TextColumn("Finish/Color", disabled=True),
            "Dimensions": st.column_config.TextColumn("Dimensions", disabled=True),
            "Product Website": st.column_config.TextColumn("Product Website", disabled=True),
            "link_text": st.column_config.TextColumn("link_text", disabled=True),
        }

        # Edits apply only when you click Save — avoids partial reruns breaking choices
        with st.form("room_editor"):
            edited_df = st.data_editor(
                df_show,
                key="links_editor",
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                column_config=col_cfg,
            )
            saved = st.form_submit_button("Save room edits", type="primary")

        if saved:
            st.session_state["extracted_df"] = edited_df
            st.success("Room edits saved.")

        st.download_button(
            "Download CSV",
            st.session_state["extracted_df"].to_csv(index=False).encode("utf-8"),
            file_name="canva_links_with_position.csv",
            mime="text/csv",
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
            url_col_guess = (
                "link_url" if "link_url" in df_in.columns
                else "Product URL" if "Product URL" in df_in.columns
                else df_in.columns[min(1, len(df_in)-1)]
            )
            url_col = st.text_input("URL column name", url_col_guess)

            # Chunk & resume controls
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                max_per_run = st.number_input(
                    "Max per run", min_value=1, max_value=2000, value=100, step=50,
                    help="Process at most this many pending rows on this click."
                )
            with c2:
                start_at = st.number_input(
                    "Skip first N pending", min_value=0, max_value=100000, value=0, step=1,
                    help="Start after this many pending rows (for manual resume)."
                )
            with c3:
                autosave_every = st.number_input(
                    "Autosave every N rows", min_value=0, max_value=1000, value=25, step=5,
                    help="0 = off. Saves a partial CSV to memory you can download if the run stops."
                )

            # Skip list + auto-skip controls
            with st.expander("Skip list / Auto-skip settings"):
                csk1, csk2 = st.columns([2,1])
                with csk1:
                    current_skip = "\n".join(st.session_state.get("skip_urls", []))
                    edited_skip = st.text_area(
                        "URLs to skip (one per line)",
                        value=current_skip,
                        height=120,
                        help="Any URL here will be skipped during enrichment."
                    )
                    if st.button("Update skip list"):
                        st.session_state["skip_urls"] = [u.strip() for u in edited_skip.splitlines() if u.strip()]
                        st.success("Skip list updated.")
                with csk2:
                    if st.session_state.get("skip_urls"):
                        skip_csv = ("\n".join(st.session_state["skip_urls"]).encode("utf-8"))
                        st.download_button(
                            "Download skip list",
                            data=skip_csv,
                            file_name="skip_urls.txt",
                            mime="text/plain",
                        )

                csk3, csk4 = st.columns([1,1])
                with csk3:
                    st.session_state["enable_auto_skip"] = st.checkbox(
                        "Enable auto-skip after N failures", value=bool(st.session_state.get("enable_auto_skip", True))
                    )
                with csk4:
                    st.session_state["auto_skip_after_n"] = st.number_input(
                        "N failures", min_value=1, max_value=10, value=int(st.session_state.get("auto_skip_after_n", 2)), step=1
                    )

                if st.session_state.get("fail_counts"):
                    st.write("Failure counts (this session):", st.session_state["fail_counts"])

            # Autosave download if available
            if st.session_state.get("last_partial_csv"):
                st.download_button(
                    "Download latest autosave",
                    data=st.session_state["last_partial_csv"],
                    file_name="links_enriched_partial.csv",
                    mime="text/csv",
                )

            if st.button("Enrich (Image URL + Price + Product Name)", key="enrich_btn"):
                with st.spinner("Scraping image + price + product name..."):
                    df_out = enrich_urls(
                        df_in, url_col, FIRECRAWL_API_KEY,
                        max_per_run=int(max_per_run), start_at=int(start_at), autosave_every=int(autosave_every)
                    )
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

# --- Tab 3: Test a single URL ---
with tab3:
    st.caption("Paste a single product URL and test the enrichment (Firecrawl v2 first, then fallback).")
    test_url = st.text_input(
        "Product URL to test",
        "https://www.lumens.com/vishal-chandelier-by-troy-lighting-TRY2622687.html"
    )
    if st.button("Run test", key="single_test_btn"):
        img = price = title = ""; status = ""
        if FIRECRAWL_API_KEY:
            if "lumens.com" in test_url:
                img, price, title, status = enrich_lumens_v2(test_url, FIRECRAWL_API_KEY)
            elif "fergusonhome.com" in test_url:
                img, price, title, status = enrich_ferguson_v2(test_url, FIRECRAWL_API_KEY)
            elif "wayfair.com" in test_url:
                img, price, title, status = enrich_wayfair_v2(test_url, FIRECRAWL_API_KEY)
            else:
                img, price, title, status = enrich_domain_firecrawl_v2(test_url, FIRECRAWL_API_KEY)

        if not img or not price or not title:
            r = requests_get(test_url)
            if r and r.text:
                i2, p2, t2 = pick_image_and_price_bs4(r.text, test_url)
                img = img or _first_scalar(i2)
                price = price or _first_scalar(p2)
                title = title or normalize_product_title(_first_scalar(t2), test_url)
                status = (status + "+bs4_ok") if status else "bs4_ok"
            else:
                status = (status + "+fetch_failed") if status else "fetch_failed"

        st.write("**Status:**", status or "unknown")
        st.write("**Image URL:**", img or "—")
        st.write("**Price:**", price or "—")
        st.write("**Product Name:**", title or "NAME NEEDED")
        if img:
            st.image(img, caption="Preview", use_container_width=True)








