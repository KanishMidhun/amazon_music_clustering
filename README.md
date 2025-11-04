# 🎵 Amazon Music Clustering — Genre & Mood Segmentation

Automatically group songs into genres or moods based on their **audio features** using unsupervised machine learning.
This project applies **K-Means Clustering**, **PCA**, and **visualization techniques** to uncover patterns in Amazon Music’s dataset.

---

## 🚀 Project Overview

With millions of songs available on streaming platforms, manually labeling them into genres or moods is not scalable.
This project clusters songs based on **audio characteristics** like tempo, energy, danceability, and acousticness — revealing meaningful groupings such as *party tracks*, *chill acoustic songs*, or *instrumental ambient tracks*.

---

## 🧠 Skills & Concepts

* Exploratory Data Analysis (EDA)
* Feature Scaling (StandardScaler)
* K-Means Clustering
* PCA for Dimensionality Reduction
* Cluster Evaluation (Silhouette, Davies–Bouldin)
* Cluster Profiling
* Streamlit Dashboard
* Data Visualization (Matplotlib, Seaborn)

---

## 📊 Dataset

**File:** `single_genre_artists.csv`
**Features used:**

| Feature          | Description                              |
| ---------------- | ---------------------------------------- |
| danceability     | How suitable a track is for dancing      |
| energy           | Intensity and activity of a track        |
| loudness         | Overall loudness in decibels             |
| speechiness      | Presence of spoken words                 |
| acousticness     | Confidence measure of acoustic sound     |
| instrumentalness | Likelihood of no vocals                  |
| liveness         | Presence of audience or live performance |
| valence          | Musical positiveness                     |
| tempo            | Beats per minute                         |
| duration_ms      | Track duration in milliseconds           |

Removed columns: `track_id`, `track_name`, `artist_name` (used only for reference).

---

## 🧩 Project Pipeline

### 1️⃣ Data Preprocessing

* Handle missing values and duplicates.
* Select relevant numeric audio features.
* Apply **StandardScaler** for normalization.
* Visualize distributions before and after scaling.

### 2️⃣ Feature Selection

* Focus on core musical descriptors that define rhythm, energy, and mood.

### 3️⃣ Clustering (K-Means)

* Test `k` values from 2–10.
* Evaluate using:

  * **Silhouette Score**
  * **Davies–Bouldin Index**
  * **Inertia (Elbow Method)**
* Best result: **k = 3**

### 4️⃣ Cluster Profiling

| Cluster | Description               | Characteristics                |
| ------- | ------------------------- | ------------------------------ |
| 0       | Party / Upbeat 🎉         | High energy, high danceability |
| 1       | Chill Acoustic 🌙         | High acousticness, low energy  |
| 2       | Instrumental / Ambient 🎧 | High instrumentalness          |

### 5️⃣ Dimensionality Reduction & Visualization

* Applied **PCA** for 2D visualization.
* Color-coded clusters.
* Created heatmaps of mean feature values per cluster.

### 6️⃣ Output

* Added cluster labels and mood descriptions.
* Exported final dataset as:

  ```
  single_genre_artists_kmeans_k3.csv
  ```

---

## 💻 Streamlit App

A fully interactive dashboard allows you to:

* Upload dataset
* View EDA summaries
* Visualize PCA plots
* Inspect cluster statistics
* Download final CSV with mood labels

### ▶️ Run the app

```bash
streamlit run amazon_music.py
```


## 🧮 Evaluation Metrics

| Metric                   | Description                       |
| ------------------------ | --------------------------------- |
| **Silhouette Score**     | Measures cohesion within clusters |
| **Davies–Bouldin Index** | Lower = better separation         |
| **Inertia**              | Compactness of clusters           |

**Best model:**
`K = 3 → Silhouette = 0.2431, DB Index = 1.5716`

---

## 📈 Visual Outputs

* PCA scatter plot (2D)
* Feature heatmaps per cluster
* Distribution plots for `energy`, `danceability`, etc.

---

## 🧩 Tech Stack

| Category      | Tools Used                                               |
| ------------- | -------------------------------------------------------- |
| Programming   | Python 3.10+                                             |
| Libraries     | pandas, numpy, scikit-learn, seaborn, matplotlib, joblib |
| Visualization | matplotlib, seaborn, Streamlit                           |
| Deployment    | Streamlit Dashboard                                      |
| Output        | CSV, PPTX (Presentation)                                 |



## 🏁 Results Summary

✅ 3 clusters successfully identified
🎧 Captured meaningful moods from numerical audio data
📊 Delivered interpretable visualization and Streamlit app
💡 Foundation for music recommendation or genre tagging systems

---
