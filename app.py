import os, io, re, json, time, requests
from typing import Optional, Tuple
import streamlit as st
import fitz  # PyMuPDF
import pandas as pd

st.set_page_config(page_title="Spec Link Extractor & Enricher", layout="wide")
st.title("🧰 Spec Link Extractor & Enricher")

ROOM_CHOICES = [
    "Plumbing", "Lighting", "Tile + Stone", "Countertops + Slabs", "Doors, Base, Case",
    "Wall Coverings", "Paint", "Cabinetry Finishes", "Hardware", "Accent Mirrors",
    "Appliances", "Other Materials", "Unassigned"
]

def extract_links_by_pages(pdf_bytes, page_to_tag, only_listed_pages=True):
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
            rows.append({
                "page": pidx,
                "Tags": tag_value,
                "Room": "Unassigned",
                "Type": "",
                "link_url": uri
            })
    return pd.DataFrame(rows)

with st.sidebar:
    st.subheader("Firecrawl (optional)")
    api_key_input = st.text_input("FIRECRAWL_API_KEY", value="", type="password")

tab1, tab2, tab3 = st.tabs(["1) Extract from PDF", "2) Enrich CSV", "3) Test URL"])

with tab1:
    st.caption("Extract links and assign rooms.")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")

    mapping_df = st.data_editor(pd.DataFrame([{ "page": "", "Tags": "" }]), num_rows="dynamic", use_container_width=True, key="page_tag_editor")
    only_listed = st.checkbox("Only extract listed pages", value=True)

    if "spec_df" not in st.session_state:
        st.session_state["spec_df"] = pd.DataFrame()

    if st.button("Extract / Refresh Table", disabled=(pdf_file is None)) and pdf_file:
        page_to_tag = {}
        _peek = fitz.open("pdf", pdf_file.getvalue())
        num_pages = len(_peek)
        for _, r in mapping_df.iterrows():
            try:
                p = int(str(r.get("page", "")).strip())
                if 1 <= p <= num_pages:
                    page_to_tag[p] = str(r.get("Tags", ""))
            except:
                continue
        pdf_bytes = pdf_file.read()
        df = extract_links_by_pages(pdf_bytes, page_to_tag, only_listed_pages=only_listed)
        st.session_state["spec_df"] = df.reset_index(drop=True)
        st.success(f"Extracted {len(df)} rows.")

    if not st.session_state["spec_df"].empty:
        raw_latest = st.session_state.get("extracted_links_editor", st.session_state["spec_df"])
        if isinstance(raw_latest, (list, dict)):
            try:
                latest_df = pd.DataFrame.from_records(raw_latest)
            except Exception:
                latest_df = st.session_state["spec_df"].copy()
        elif isinstance(raw_latest, pd.DataFrame):
            latest_df = raw_latest.copy()
        else:
            latest_df = st.session_state["spec_df"].copy()

        latest_df = latest_df.reset_index(drop=True)

        edited_df = st.data_editor(
            latest_df,
            use_container_width=True,
            key="extracted_links_editor",
            column_config={
                "Room": st.column_config.SelectboxColumn("Room", options=ROOM_CHOICES)
            }
        )

        st.session_state["spec_df"] = edited_df.reset_index(drop=True)

        st.download_button(
            "Download CSV",
            edited_df.to_csv(index=False).encode('utf-8'),
            file_name="links.csv",
            mime="text/csv"
        )
