# =============================================================
# 🎵 AMAZON MUSIC CLUSTERING — STREAMLIT APP (K-MEANS ONLY)
# =============================================================
import os
import io
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

sns.set(style="whitegrid")
st.set_page_config(page_title="Amazon Music Clustering (K-Means)", layout="wide")

# =============================================================
# Utility functions
# =============================================================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

@st.cache_data
def run_pca(X_scaled):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled)

def create_description_map(profile_df):
    norm = (profile_df - profile_df.min()) / (profile_df.max() - profile_df.min())
    desc_map = {}
    for cid, row in norm.iterrows():
        high = row[row > 0.7].index.tolist()
        low = row[row < 0.3].index.tolist()
        if 'energy' in high and 'danceability' in high:
            mood = "Party / Upbeat 🎉"
        elif 'acousticness' in high and 'energy' in low:
            mood = "Chill Acoustic 🌙"
        elif 'instrumentalness' in high:
            mood = "Instrumental / Ambient 🎧"
        elif 'speechiness' in high:
            mood = "Speech-heavy 🎤"
        else:
            mood = "Balanced / Mixed 🎵"
        desc_map[cid] = mood
    return desc_map

def df_to_csv_bytes(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

# =============================================================
# Streamlit UI
# =============================================================
st.title("🎶 Amazon Music Clustering — K-Means Only")
st.markdown("Unsupervised grouping of songs using K-Means clustering and PCA visualization.")

DATA_PATH = st.sidebar.text_input("📂 Enter CSV Path", value="single_genre_artists.csv")

features = [
    'danceability','energy','loudness','speechiness','acousticness',
    'instrumentalness','liveness','valence','tempo','duration_ms'
]

st.sidebar.markdown("### 🔹 Choose K-Means Parameters")
k_min = st.sidebar.slider("Minimum k (for elbow)", 2, 5, 3)
k_max = st.sidebar.slider("Maximum k (for elbow)", 6, 20, 10)
k_final = st.sidebar.slider("Final number of clusters (k)", 2, 15, 6)
run = st.sidebar.button("🚀 Run K-Means Clustering")

# =============================================================
# Main Logic
# =============================================================
if run:
    start = time.time()
    if not os.path.exists(DATA_PATH):
        st.error("❌ Invalid CSV path. Please provide a valid file.")
    else:
        df = load_data(DATA_PATH)
        st.subheader("📊 Data Preview")
        st.write(df.head())
        st.write("Shape:", df.shape)

        drop_cols = [c for c in ['track_id','track_name','artist_name','id_songs','id_artists','name_song','name_artists'] if c in df.columns]
        df = df.drop(columns=drop_cols, errors='ignore')
        df = df[features]

        # Scaling
        X_scaled, scaler = scale_features(df)
        st.info(f"Data scaled. Shape: {X_scaled.shape}")

        # =========================================================
        # 🔍 Elbow Method
        # =========================================================
        st.markdown("## 📈 Elbow Method — Finding Optimal k")
        inertias = []
        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        fig_elbow = plt.figure(figsize=(6,4))
        plt.plot(range(k_min, k_max + 1), inertias, 'bo-')
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (SSE)")
        plt.title("Elbow Method")
        st.pyplot(fig_elbow)

        st.caption("Look for the 'elbow point' where the drop in inertia slows down — that’s your ideal k.")

        # =========================================================
        # 🚀 K-Means (Final)
        # =========================================================
        st.markdown("## 🤖 K-Means Clustering Results")
        kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
        k_labels = kmeans.fit_predict(X_scaled)
        df['kmeans_cluster'] = k_labels

        sil = silhouette_score(X_scaled, k_labels)
        dbi = davies_bouldin_score(X_scaled, k_labels)
        st.success(f"Silhouette = {sil:.4f} | Davies–Bouldin = {dbi:.4f}")

        # PCA visualization
        X_pca = run_pca(X_scaled)
        fig_pca = plt.figure(figsize=(7,5))
        plt.scatter(X_pca[:,0], X_pca[:,1], c=k_labels, cmap='tab10', s=10, alpha=0.7)
        plt.title(f"PCA Visualization — K-Means (k={k_final})")
        st.pyplot(fig_pca)

        # =========================================================
        # 📊 Cluster Profiling
        # =========================================================
        st.markdown("## 📊 Cluster Profiles (Mean Feature Values)")
        profile = df.groupby('kmeans_cluster')[features].mean().round(3)
        desc_map = create_description_map(profile)
        df['cluster_description'] = df['kmeans_cluster'].map(desc_map)

        st.dataframe(profile)
        st.markdown("### 🗂️ Cluster Descriptions")
        st.dataframe(pd.DataFrame(list(desc_map.items()), columns=['Cluster','Description']))

        # Heatmap
        fig_heat, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(profile.T, cmap="coolwarm", annot=True, ax=ax)
        plt.title("Cluster Feature Averages (K-Means)")
        st.pyplot(fig_heat)

        # =========================================================
        # 💾 Export
        # =========================================================
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        out_csv = os.path.join("outputs", f"music_kmeans_k{k_final}.csv")
        df.to_csv(out_csv, index=False)
        joblib.dump(kmeans, os.path.join("models", f"kmeans_k{k_final}.joblib"))
        joblib.dump(scaler, os.path.join("models", "standard_scaler.joblib"))

        st.success(f"✅ Results saved to {out_csv}")
        st.download_button(
            "📥 Download Clustered CSV",
            data=df_to_csv_bytes(df),
            file_name=f"music_kmeans_k{k_final}.csv",
            mime="text/csv"
        )

        elapsed = time.time() - start
        st.info(f"⏱️ Runtime: {elapsed:.2f} seconds")
else:
    st.info("👈 Choose your dataset and click 'Run K-Means Clustering' to begin.")
