# Fixed unmatched parentheses and duplicate code blocks
# Cleaned to ensure Tab1 runs once without duplicated sections causing syntax error.

import os, io, re, json, time, requests
from typing import Optional, Tuple
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

st.set_page_config(page_title="Spec Link Extractor & Enricher", layout="wide")
st.title("🧰 Spec Link Extractor & Enricher")

# --- Timer ---
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t0

# --- Regex ---
QTY_RE = re.compile(r"(?i)\\b(?:QTY|Quantity)\\b\\s*[:\\-]?\\s*([0-9X]+)\\b")
FINISH_RE = re.compile(r"(?i)\\bFinish\\b\\s*[:\\-]?\\s*(.+)")
SIZE_RE = re.compile(r"(?i)\\b(?:Size|Dimensions?)\\b\\s*[:\\-]?\\s*(.+)")
TYPE_RE = re.compile(r"(?i)\\bType\\b\\s*[:\\-]?\\s*(.+)")

POS_AT_START = re.compile(r'^\\s*(\\d{1,3})[.\\u2024\\u00B7]?\\s*')
NEXT_BULLET = re.compile(r'\\s(\\d{1,3})[.\\u2024\\u00B7](?=\\s|[A-Z])')

ROOM_CHOICES = [
    "Plumbing","Lighting","Tile + Stone","Countertops + Slabs","Doors, Base, Case",
    "Wall Coverings","Paint","Cabinetry Finishes","Hardware","Accent Mirrors",
    "Appliances","Other Materials","Unassigned"
]

# --- Helper Functions ---
def _normalize_separators(s):
    if not s: return ""
    s = s.replace("\n", " | ")
    s = re.sub(r"\\s*\\|\\s*", " | ", s)
    return s.strip()

def parse_link_title_fields(text):
    fields = {"Type": "", "QTY": "", "Finish": "", "Size": ""}
    s = _normalize_separators(text)
    parts = [p.strip() for c in s.split('|') for p in c.split(';') if p.strip()]
    for tok in parts:
        for name, rx in [("Type", TYPE_RE),("QTY", QTY_RE),("Finish", FINISH_RE),("Size", SIZE_RE)]:
            if not fields[name]:
                m = rx.search(tok)
                if m:
                    fields[name] = m.group(1).strip()
                    break
    if not fields["Type"] and parts:
        for tok in parts:
            if ":" not in tok and "-" not in tok:
                fields["Type"] = tok.strip(); break
    return fields

def extract_links_by_pages(pdf_bytes, page_to_tag, only_listed_pages=True, pad_px=4.0, band_px=28.0):
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
            rows.append({"page":pidx,"Tags":tag_value,"Room":"Unassigned","Type":"","link_url":uri})
    return pd.DataFrame(rows)

# --- Sidebar ---
with st.sidebar:
    st.subheader("Firecrawl (optional)")
    api_key_input = st.text_input("FIRECRAWL_API_KEY", value="", type="password")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["1) Extract from PDF","2) Enrich CSV","3) Test URL"])

# --- Tab 1 ---
with tab1:
    st.caption("Extract links and assign rooms.")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    num_pages = None
    if pdf_file:
        try:
            _peek = fitz.open("pdf", pdf_file.getvalue())
            num_pages = len(_peek)
            st.info(f"PDF detected with {num_pages} page(s).")
        except Exception as e:
            st.error(str(e))

    mapping_df = st.data_editor(pd.DataFrame([{"page":"","Tags":""}]), num_rows="dynamic", use_container_width=True, key="page_tag_editor")
    only_listed = st.checkbox("Only extract listed pages", value=True)

    if "spec_df" not in st.session_state:
        st.session_state["spec_df"] = pd.DataFrame()

    if st.button("Extract / Refresh Table", disabled=(pdf_file is None)) and pdf_file:
        page_to_tag = {}
        if num_pages is None:
            num_pages = len(fitz.open("pdf", pdf_file.getvalue()))
        for _, r in mapping_df.iterrows():
            try:
                p = int(str(r.get("page","")).strip())
                page_to_tag[p] = str(r.get("Tags",""))
            except: continue
        pdf_bytes = pdf_file.read()
        df = extract_links_by_pages(pdf_bytes, page_to_tag, only_listed_pages=only_listed)
        if df.empty:
            st.info("No links found.")
        else:
            st.session_state["spec_df"] = df.reset_index(drop=True)
            st.success(f"Extracted {len(df)} rows.")

    if not st.session_state["spec_df"].empty:
        latest_df = st.session_state.get("extracted_links_editor", st.session_state["spec_df"])
        if isinstance(latest_df, dict): latest_df = pd.DataFrame(latest_df)
        edited_df = st.data_editor(latest_df.reset_index(drop=True), use_container_width=True, key="extracted_links_editor", column_config={"Room": st.column_config.SelectboxColumn("Room", options=ROOM_CHOICES)})
        st.session_state["spec_df"] = edited_df.reset_index(drop=True)
        st.download_button("Download CSV", edited_df.to_csv(index=False).encode('utf-8'), file_name="links.csv", mime="text/csv")
