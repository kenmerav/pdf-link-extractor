if st.button("Enrich (Image URL + Price)", key="enrich_btn"):
                with st.spinner("Scraping image + price..."):
                    df_out = enrich_urls(df_in, url_col, api_key_input)
                # Coerce problematic columns to plain strings to avoid Arrow list/non-list issues
                for c in ["scraped_image_url", "price", "scrape_status"]:
                    if c in df_out.columns:
                        df_out[c] = df_out[c].apply(_first_scalar).astype(str).fillna("")
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
