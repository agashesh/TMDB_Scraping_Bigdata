import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prediksi Revenue Film", layout="wide")

FITUR_X = ["budget", "runtime", "popularity", "vote_average", "vote_count"]


@st.cache_data
def load_data():
    df = pd.read_csv("movies_dataset.csv")
    return df


df_raw = load_data()

st.title("🎬 Dashboard Prediksi Revenue Film")
st.caption("Big Data & Predictive Analytics — Final Project | Sumber data: TMDB API")

# ---------------- SIDEBAR FILTER ----------------
st.sidebar.header("Filter Data")
df = df_raw.copy()

if "release_date" in df.columns:
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    year_range = st.sidebar.slider("Tahun Rilis", min_year, max_year, (min_year, max_year))
    df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

if "genres" in df.columns:
    all_genres = sorted({
        g.strip() for sub in df["genres"].dropna().str.split(",") for g in sub if g.strip()
    })
    selected_genres = st.sidebar.multiselect("Genre", all_genres)
    if selected_genres:
        df = df[df["genres"].apply(
            lambda x: any(g in x for g in selected_genres) if pd.notna(x) else False
        )]

st.sidebar.markdown("---")
st.sidebar.caption(f"Menampilkan {len(df):,} dari {len(df_raw):,} film")

tab1, tab2, tab3 = st.tabs(["📊 Data Summary", "📈 Visualisasi", "🔮 Prediksi Revenue"])

# ---------------- TAB 1: DATA SUMMARY ----------------
with tab1:
    st.subheader("Ringkasan Data")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Film", f"{len(df):,}")
    col2.metric("Rata-rata Budget", f"${df['budget'].mean():,.0f}")
    col3.metric("Rata-rata Revenue", f"${df['revenue'].mean():,.0f}")
    col4.metric("Rata-rata Rating", f"{df['vote_average'].mean():.2f}")

    st.markdown("**Tabel Data (diurutkan berdasarkan Revenue tertinggi)**")
    kolom_tampil = [c for c in
                    ["title", "budget", "revenue", "runtime", "popularity", "vote_average", "vote_count", "genres"]
                    if c in df.columns]
    st.dataframe(
        df[kolom_tampil].sort_values("revenue", ascending=False),
        use_container_width=True,
        height=420,
    )

# ---------------- TAB 2: VISUALISASI ----------------
with tab2:
    st.subheader("Correlation Heatmap")
    fitur_corr = [c for c in ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"] if c in df.columns]
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[fitur_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax1)
    ax1.set_title("Correlation Heatmap Antar Variabel")
    st.pyplot(fig1)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Scatter Plot: Budget vs Revenue")
        fig2, ax2 = plt.subplots(figsize=(5.5, 4))
        ax2.scatter(df["budget"], df["revenue"], alpha=0.5, color="#3b5bdb")
        ax2.set_xlabel("Budget (USD)")
        ax2.set_ylabel("Revenue (USD)")
        ax2.set_title("Budget vs Revenue")
        st.pyplot(fig2)

    with col_b:
        st.subheader("Distribusi Revenue")
        fig3, ax3 = plt.subplots(figsize=(5.5, 4))
        sns.histplot(df["revenue"], bins=30, kde=True, ax=ax3, color="#5c7cfa")
        ax3.set_xlabel("Revenue (USD)")
        ax3.set_title("Distribusi Revenue Film")
        st.pyplot(fig3)

    if "genres" in df.columns:
        st.subheader("Top 10 Genre Berdasarkan Rata-rata Revenue")
        genre_expanded = df.assign(genre=df["genres"].str.split(",")).explode("genre")
        genre_expanded["genre"] = genre_expanded["genre"].str.strip()
        top_genre = (
            genre_expanded.groupby("genre")["revenue"].mean()
            .sort_values(ascending=False).head(10)
        )
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.barh(top_genre.index[::-1], top_genre.values[::-1], color="#4263eb")
        ax4.set_xlabel("Rata-rata Revenue (USD)")
        ax4.set_title("Top 10 Genre - Rata-rata Revenue")
        st.pyplot(fig4)

# ---------------- TAB 3: PREDIKSI ----------------
with tab3:
    st.subheader("Model Prediksi Revenue (Multiple Linear Regression)")

    data_model = df.dropna(subset=FITUR_X + ["revenue"])
    X = data_model[FITUR_X]
    y = data_model["revenue"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    pred_test = model.predict(X_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{r2_score(y_test, pred_test):.3f}")
    col2.metric("MAE", f"${mean_absolute_error(y_test, pred_test):,.0f}")
    col3.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, pred_test)):,.0f}")

    st.markdown("---")
    st.markdown("**Coba prediksi revenue film baru:**")
    c1, c2 = st.columns(2)
    with c1:
        input_budget = st.number_input("Budget (USD)", min_value=0, value=50_000_000, step=1_000_000)
        input_runtime = st.number_input("Runtime (menit)", min_value=0, value=110)
        input_popularity = st.number_input("Popularity", min_value=0.0, value=20.0)
    with c2:
        input_vote_avg = st.slider("Vote Average", 0.0, 10.0, 7.0)
        input_vote_count = st.number_input("Vote Count", min_value=0, value=1000)

    if st.button("Prediksi Revenue", type="primary"):
        input_df = pd.DataFrame(
            [[input_budget, input_runtime, input_popularity, input_vote_avg, input_vote_count]],
            columns=FITUR_X,
        )
        hasil_prediksi = model.predict(input_df)[0]
        st.success(f"💰 Prediksi Revenue: ${hasil_prediksi:,.0f}")

    st.markdown("---")
    st.subheader("Prediksi vs Aktual (Data Uji)")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.scatter(y_test, pred_test, alpha=0.5, color="#37b24d")
    lims = [min(y_test.min(), pred_test.min()), max(y_test.max(), pred_test.max())]
    ax5.plot(lims, lims, color="red", linestyle="--", label="Prediksi Sempurna")
    ax5.set_xlabel("Revenue Aktual (USD)")
    ax5.set_ylabel("Revenue Prediksi (USD)")
    ax5.set_title("Perbandingan Revenue Aktual vs Prediksi")
    ax5.legend()
    st.pyplot(fig5)
