import streamlit as st
import fitz  # PyMuPDF
import pandas as pd

def extract_links_from_pdf(pdf_bytes):
    """Extract links + metadata from uploaded PDF file (bytes)."""
    doc = fitz.open("pdf", pdf_bytes)
    rows = []

    for page in doc:
        # Header area (right/top 10% of page)
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

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="PDF Link Extractor", layout="wide")
st.title("📄 PDF Link Extractor")
st.caption("Upload a PDF → extract all web links with project/sheet info → download a CSV.")

uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    st.info(f"File uploaded: {uploaded_pdf.name}")
    if st.button("Extract Links"):
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
else:
    st.info("Upload a PDF to begin.")
