import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from db import (
    init_db, get_draw_count, get_position_count, get_feature_count,
    get_draws_without_positions, get_draws_without_features,
    load_draws_df, load_features_df, load_positions_df, clear_all_data,
)
from ingest_powerball import ingest_from_csv
from ephemeris import EphemerisEngine
from features import generate_bulk, BIN_LABELS
from analysis import feature_number_scan

st.set_page_config(
    page_title="Powerball Planet Correlation Analyzer",
    page_icon="🪐",
    layout="wide",
)

init_db()

st.title("🪐 Powerball Planet Correlation Analyzer")
st.caption("Analyze historical Powerball draws against planetary positions at each draw date/time and location.")

with st.sidebar:
    st.header("Configuration")

    st.subheader("Draw Location")
    location_name = st.text_input("Location name", value="Sydney, Australia")
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input("Latitude", value=-33.8688, format="%.4f", min_value=-90.0, max_value=90.0)
    with col2:
        longitude = st.number_input("Longitude", value=151.2093, format="%.4f", min_value=-180.0, max_value=180.0)
    altitude_m = st.number_input("Altitude (m)", value=0.0, min_value=0.0, max_value=9000.0)
    timezone_str = st.text_input("Draw timezone", value="Australia/Sydney")

    st.divider()

    st.subheader("Feature Settings")
    bin_size = st.selectbox("Longitude bin size (degrees)", [30, 15, 10], index=0)
    orb_deg = st.slider("Aspect orb (degrees)", min_value=1.0, max_value=15.0, value=6.0, step=0.5)

    st.divider()

    st.subheader("Number Range")
    st.caption("Set based on your Powerball variant")
    number_max = st.selectbox("Max main ball number", [35, 69], index=0,
                              help="AU Powerball: 35, US Powerball: 69")
    pb_max = st.selectbox("Max Powerball number", [20, 26], index=0,
                          help="AU Powerball: 20, US Powerball: 26")

    st.divider()

    st.subheader("Database")
    draw_count = get_draw_count()
    pos_count = get_position_count()
    feat_count = get_feature_count()
    st.metric("Draws imported", draw_count)
    st.metric("Positions computed", pos_count)
    st.metric("Features generated", feat_count)

    if st.button("Clear all data", type="secondary"):
        clear_all_data()
        st.rerun()


tab_import, tab_compute, tab_analyze, tab_explore, tab_notes = st.tabs([
    "1. Import Draws", "2. Compute Positions & Features",
    "3. Correlation Analysis", "4. Explore Data", "5. Notes & Methodology"
])

with tab_import:
    st.header("Import Historical Powerball Draws")

    st.markdown("""
    Upload a CSV file with historical draw data. The CSV must contain these columns:

    **Australian Powerball (7 balls):**
    `draw_datetime_local, n1, n2, n3, n4, n5, n6, n7, powerball`

    **US Powerball (5 balls):**
    `draw_datetime_local, n1, n2, n3, n4, n5, powerball`

    The datetime should include timezone info (e.g., `2023-01-05T20:30:00+11:00`).
    If no timezone is specified, the draw timezone from the sidebar will be used.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        st.write(f"**File:** {uploaded_file.name}")
        try:
            preview_df = pd.read_csv(uploaded_file)
            st.dataframe(preview_df.head(10), use_container_width=True)
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

        if st.button("Import CSV", type="primary"):
            with st.spinner("Importing draws..."):
                try:
                    uploaded_file.seek(0)
                    result = ingest_from_csv(uploaded_file, timezone_str=timezone_str)
                    st.success(
                        f"Import complete: {result['inserted']} inserted, "
                        f"{result['skipped']} skipped (duplicates), "
                        f"{len(result['errors'])} errors. Format: {result['format']}"
                    )
                    if result["errors"]:
                        with st.expander(f"Show {len(result['errors'])} error(s)"):
                            for err in result["errors"][:50]:
                                st.warning(f"Row {err['row']}: {', '.join(err['errors'])}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    st.divider()

    if draw_count > 0:
        st.subheader("Imported Draws Preview")
        draws_df = load_draws_df()
        display_df = draws_df.copy()
        display_df["numbers"] = display_df["numbers_json"].apply(lambda s: ", ".join(str(x) for x in json.loads(s)))
        st.dataframe(
            display_df[["draw_id", "draw_datetime_local", "numbers", "powerball", "game"]].head(20),
            use_container_width=True,
        )
        st.caption(f"Showing first 20 of {len(draws_df)} draws")


with tab_compute:
    st.header("Compute Planetary Positions & Features")

    st.markdown(f"""
    **Location:** {location_name} ({latitude:.4f}, {longitude:.4f})
    | **Bin size:** {bin_size}° | **Aspect orb:** {orb_deg}°
    """)

    col_pos, col_feat = st.columns(2)

    with col_pos:
        st.subheader("Step 1: Planetary Positions")
        missing_positions = get_draws_without_positions()
        st.info(f"{len(missing_positions)} draws need position computation")

        if len(missing_positions) > 0:
            if st.button("Compute Planetary Positions", type="primary"):
                engine = EphemerisEngine()
                progress_bar = st.progress(0, text="Computing positions...")
                status_text = st.empty()

                def pos_progress(done, total):
                    progress_bar.progress(done / total, text=f"Computing positions... {done}/{total}")

                try:
                    computed = engine.compute_bulk(
                        missing_positions, latitude, longitude, altitude_m,
                        location_name, progress_callback=pos_progress
                    )
                    progress_bar.progress(1.0, text="Done!")
                    st.success(f"Computed positions for {computed} draws")
                    st.rerun()
                except Exception as e:
                    st.error(f"Computation failed: {e}")
        else:
            st.success("All draws have computed positions")

    with col_feat:
        st.subheader("Step 2: Alignment Features")
        missing_features = get_draws_without_features()
        st.info(f"{len(missing_features)} draws need feature generation")

        if len(missing_features) > 0:
            if st.button("Generate Alignment Features", type="primary"):
                progress_bar2 = st.progress(0, text="Generating features...")

                def feat_progress(done, total):
                    progress_bar2.progress(done / total, text=f"Generating features... {done}/{total}")

                try:
                    computed = generate_bulk(
                        missing_features, bin_size=bin_size, orb_deg=orb_deg,
                        progress_callback=feat_progress
                    )
                    progress_bar2.progress(1.0, text="Done!")
                    st.success(f"Generated features for {computed} draws")
                    st.rerun()
                except Exception as e:
                    st.error(f"Feature generation failed: {e}")
        else:
            st.success("All positioned draws have computed features")


with tab_analyze:
    st.header("Statistical Correlation Analysis")

    if feat_count == 0:
        st.warning("No features computed yet. Go to the Compute tab first.")
    else:
        st.markdown("""
        This analysis tests whether any planetary alignment feature is statistically
        associated with specific drawn numbers using a two-proportion z-test,
        corrected for multiple comparisons via **Benjamini-Hochberg** FDR.
        """)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            q_threshold = st.number_input("Max q-value (BH)", value=0.25, min_value=0.0, max_value=1.0, step=0.05)
        with col_f2:
            min_lift = st.number_input("Min absolute lift", value=0.0, min_value=0.0, max_value=10.0, step=0.1)
        with col_f3:
            min_group = st.number_input("Min group size", value=5, min_value=1, max_value=100, step=1)

        if st.button("Run Analysis", type="primary"):
            with st.spinner("Running correlation scan..."):
                try:
                    draws_df = load_draws_df()
                    feats_df = load_features_df()

                    results_df = feature_number_scan(
                        draws_df, feats_df,
                        number_min=1, number_max=number_max,
                        pb_max=pb_max, min_group_size=min_group,
                    )

                    if results_df.empty:
                        st.info("No testable feature-number associations found. Try adjusting settings.")
                    else:
                        st.session_state["analysis_results"] = results_df
                        st.success(f"Scan complete: {len(results_df)} associations tested")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if "analysis_results" in st.session_state:
            results_df = st.session_state["analysis_results"]

            filtered = results_df[
                (results_df["q_value_bh"] <= q_threshold) &
                (results_df["lift"].abs() >= min_lift)
            ].copy()

            st.subheader(f"Results: {len(filtered)} associations pass filters (of {len(results_df)} tested)")

            if not filtered.empty:
                st.dataframe(
                    filtered[["feature", "number", "n1", "n0", "rate1", "rate0",
                              "lift", "z_score", "p_value", "q_value_bh"]].round(6),
                    use_container_width=True,
                    height=400,
                )

                csv_data = filtered.to_csv(index=False)
                st.download_button(
                    "Download filtered results as CSV",
                    csv_data,
                    file_name="correlation_results.csv",
                    mime="text/csv",
                )

                st.subheader("Top Associations Visualization")
                top_n = min(20, len(filtered))
                top = filtered.head(top_n).copy()
                top["label"] = top["feature"] + " → " + top["number"]
                top["neg_log_q"] = -np.log10(top["q_value_bh"].clip(lower=1e-300))

                fig = px.bar(
                    top, x="label", y="lift",
                    color="neg_log_q",
                    color_continuous_scale="Viridis",
                    labels={"lift": "Lift (rate ratio - 1)", "neg_log_q": "-log10(q)", "label": ""},
                    title=f"Top {top_n} Associations by p-value",
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No associations pass the current filter thresholds. Try relaxing filters.")


with tab_explore:
    st.header("Explore Data")

    if draw_count == 0:
        st.info("Import draws first to explore data.")
    else:
        explore_tab1, explore_tab2, explore_tab3 = st.tabs([
            "Draw History", "Planetary Positions", "Feature Distribution"
        ])

        with explore_tab1:
            draws_df = load_draws_df()
            draws_df["numbers"] = draws_df["numbers_json"].apply(
                lambda s: ", ".join(str(x) for x in json.loads(s))
            )

            st.subheader("Number Frequency")
            all_numbers = []
            for s in draws_df["numbers_json"]:
                all_numbers.extend(json.loads(s))

            if all_numbers:
                freq_df = pd.DataFrame({"number": all_numbers})
                fig_freq = px.histogram(freq_df, x="number", nbins=number_max,
                                        title="Main Ball Frequency Distribution",
                                        labels={"number": "Ball Number", "count": "Frequency"})
                st.plotly_chart(fig_freq, use_container_width=True)

            st.subheader("Powerball Frequency")
            pb_freq = draws_df["powerball"].value_counts().sort_index()
            fig_pb = px.bar(x=pb_freq.index, y=pb_freq.values,
                            title="Powerball Frequency Distribution",
                            labels={"x": "Powerball Number", "y": "Frequency"})
            st.plotly_chart(fig_pb, use_container_width=True)

            st.subheader("Full Draw Table")
            st.dataframe(
                draws_df[["draw_id", "draw_datetime_local", "numbers", "powerball", "game"]],
                use_container_width=True,
                height=400,
            )

        with explore_tab2:
            if pos_count == 0:
                st.info("Compute positions first.")
            else:
                pos_df = load_positions_df()
                st.subheader("Planetary Longitude Over Time")

                plot_data = []
                for _, row in pos_df.iterrows():
                    positions = json.loads(row["positions_json"])
                    draw_id = row["draw_id"]
                    for planet, coords in positions.items():
                        plot_data.append({
                            "draw_id": draw_id,
                            "planet": planet,
                            "ecl_lon": coords["ecl_lon"],
                            "ecl_lat": coords["ecl_lat"],
                        })

                if plot_data:
                    plot_df = pd.DataFrame(plot_data)
                    draws_for_merge = load_draws_df()[["draw_id", "draw_datetime_utc"]]
                    plot_df = plot_df.merge(draws_for_merge, on="draw_id")
                    plot_df["date"] = pd.to_datetime(plot_df["draw_datetime_utc"])

                    fig_lon = px.scatter(
                        plot_df, x="date", y="ecl_lon", color="planet",
                        title="Ecliptic Longitude by Planet Over Time",
                        labels={"ecl_lon": "Ecliptic Longitude (°)", "date": "Draw Date"},
                        height=500,
                    )
                    st.plotly_chart(fig_lon, use_container_width=True)

                    selected_planet = st.selectbox("Select planet for detail", sorted(plot_df["planet"].unique()))
                    planet_data = plot_df[plot_df["planet"] == selected_planet]
                    fig_detail = px.line(
                        planet_data, x="date", y="ecl_lon",
                        title=f"{selected_planet.capitalize()} Ecliptic Longitude",
                        labels={"ecl_lon": "Longitude (°)", "date": "Date"},
                    )
                    st.plotly_chart(fig_detail, use_container_width=True)

        with explore_tab3:
            if feat_count == 0:
                st.info("Generate features first.")
            else:
                feats_df = load_features_df()

                all_feat_dicts = [json.loads(row["features_json"]) for _, row in feats_df.iterrows()]
                if all_feat_dicts:
                    feat_matrix = pd.DataFrame(all_feat_dicts)
                    numeric_cols = feat_matrix.select_dtypes(include=["number"]).columns

                    st.subheader("Aspect Frequencies")
                    aspect_cols = [c for c in numeric_cols if c.startswith("aspect_")]
                    if aspect_cols:
                        aspect_sums = feat_matrix[aspect_cols].sum().sort_values(ascending=False).head(30)
                        fig_asp = px.bar(
                            x=aspect_sums.index, y=aspect_sums.values,
                            title="Top 30 Most Frequent Aspects",
                            labels={"x": "Aspect", "y": "Count"},
                        )
                        fig_asp.update_layout(xaxis_tickangle=-45, height=500)
                        st.plotly_chart(fig_asp, use_container_width=True)

                    st.subheader("Longitude Bin Distribution")
                    bin_cols = [c for c in numeric_cols if c.startswith("bin_")]
                    if bin_cols:
                        bin_sums = feat_matrix[bin_cols].sum().sort_values(ascending=False).head(30)
                        fig_bin = px.bar(
                            x=bin_sums.index, y=bin_sums.values,
                            title="Top 30 Longitude Bins (by frequency)",
                            labels={"x": "Bin", "y": "Count"},
                        )
                        fig_bin.update_layout(xaxis_tickangle=-45, height=500)
                        st.plotly_chart(fig_bin, use_container_width=True)


with tab_notes:
    st.header("Notes & Methodology")

    st.markdown("""
    ### How This Works

    This application tests whether planetary positions at the time and location of
    historical Powerball draws show any statistical association with the numbers drawn.

    ### Workflow

    1. **Import** historical Powerball draw data (CSV upload)
    2. **Compute** planetary positions using the Skyfield astronomical library (DE421 ephemeris)
    3. **Generate** alignment features from those positions
    4. **Analyze** correlations using statistical tests with multiple comparison corrections

    ### Planetary Position Calculation

    - Uses **Skyfield** with the DE421 JPL ephemeris file
    - Computes **topocentric** positions from the selected observer location
    - Returns **ecliptic longitude** and **latitude** for each planet
    - Planets included: Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Moon, Sun

    ### Feature Engineering

    **Longitude Bins:** Each planet's ecliptic longitude is divided into bins
    (default 30° = 12 zodiac signs). This creates binary features indicating
    which bin each planet occupies at each draw time.

    **Pairwise Aspects:** For every pair of planets, the angular separation is
    computed and compared against classical aspects:
    - **Conjunction** (0°) — planets at the same ecliptic longitude
    - **Sextile** (60°) — planets 60° apart
    - **Square** (90°) — planets 90° apart
    - **Trine** (120°) — planets 120° apart
    - **Opposition** (180°) — planets directly opposite

    An aspect is flagged as active if the angular separation is within the
    configured **orb** (tolerance) of the exact aspect angle.

    ### Statistical Testing

    **Two-proportion z-test:** For each feature-number combination, the test compares
    the draw rate of that number when the feature is active vs. inactive.

    **Benjamini-Hochberg correction:** Because thousands of tests are run simultaneously,
    raw p-values are corrected using the BH procedure to control the
    **false discovery rate** (FDR). The q-value represents the expected proportion
    of false positives among results at that threshold.

    ### Important Caveats

    - Lottery draws are designed to be **random and independent**
    - Any correlations found are almost certainly due to **chance** or **multiple testing**
    - This tool is for **educational and exploratory** purposes only
    - Surviving BH correction does not imply a causal relationship
    - Results should not be used for actual betting decisions
    """)
