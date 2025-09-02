import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Interactive EDA", layout="wide")

st.title("Interactive EDA: Heatmap • Scatter • Histogram")

@st.cache_data
def read_csv(uploaded):
    return pd.read_csv(uploaded)

st.sidebar.header("Load data")
uploaded = st.sidebar.file_uploader("Upload CSV (daily data)", type=["csv"])
df = None

if uploaded is not None:
    df = read_csv(uploaded)
else:
    st.info("Upload a CSV to begin. Expect a 'day' column (date) and numeric columns such as daily_count, lags, etc.")

if df is not None:
    # try to normalize date
    if "day" in df.columns:
        df["day"] = pd.to_datetime(df["day"], errors="coerce")
        df = df.sort_values("day")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found.")
        st.stop()

    tabs = st.tabs(["Heatmap", "Scatter", "Histogram"])

    # ---------- Heatmap ----------
    with tabs[0]:
        st.subheader("Correlation heatmap")
        sel = st.multiselect("Pick numeric columns", options=numeric_cols,
                             default=numeric_cols[: min(6, len(numeric_cols))])
        if len(sel) >= 2:
            corr = df[sel].dropna().corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Select at least 2 columns")

    # ---------- Scatter ----------
    with tabs[1]:
        st.subheader("Scatter")
        c1, c2, c3 = st.columns([1,1,1])
        x = c1.selectbox("X", numeric_cols, index=min(1, len(numeric_cols)-1))
        y = c2.selectbox("Y", numeric_cols, index=0)
        sample_n = c3.slider("Sample (rows)", min_value=500, max_value=min(10000, len(df)), value=min(3000, len(df)), step=500)
        d = df[[x, y]].dropna()
        if len(d) > sample_n:
            d = d.sample(sample_n, random_state=42)
        fig = px.scatter(d, x=x, y=y, opacity=0.75, hover_data=d.columns, title=f"{y} vs {x}")
        fig.update_traces(marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Histogram ----------
    with tabs[2]:
        st.subheader("Histogram")
        col = st.selectbox("Column", numeric_cols, index=0)
        bins = st.slider("Bins", min_value=10, max_value=120, value=30, step=5)
        fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
        fig.update_xaxes(rangeslider=dict(visible=True))
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Tip: Use the legend to hide/show traces and the range slider on histograms to zoom.")
