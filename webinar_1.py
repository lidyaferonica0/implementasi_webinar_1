
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

np.random.seed(42)

profil = [
    {
        "jumlah": 40,
        "jam_belajar": (10, 3),
        "kelompok": (4, 2),
        "perpustakaan": (6, 2),
        "online": (3, 1.5),
    },
    {
        "jumlah": 40,
        "jam_belajar": (7, 2),
        "kelompok": (3, 2),
        "perpustakaan": (3, 2),
        "online": (6, 2),
    },
    {
        "jumlah": 40,
        "jam_belajar": (3.5, 1.5),
        "kelompok": (1, 1),
        "perpustakaan": (1, 1),
        "online": (8, 2),
    },
]

data_list = []

for p in profil:
    n = p["jumlah"]

    data_list.append(
        pd.DataFrame({
            "jam_belajar_per_minggu": np.clip(
                np.random.normal(*p["jam_belajar"], n), 0, 30
            ),
            "belajar_kelompok_per_bulan": np.clip(
                np.round(np.random.normal(*p["kelompok"], n)), 0, 15
            ),
            "kunjungan_perpustakaan_per_bulan": np.clip(
                np.round(np.random.normal(*p["perpustakaan"], n)), 0, 20
            ),
            "jam_belajar_online_per_minggu": np.clip(
                np.random.normal(*p["online"], n), 0, 30
            ),
        })
    )

df = pd.concat(data_list, ignore_index=True)
df.insert(0, "id_mahasiswa", [
    f"MHS{i:03d}" for i in range(1, len(df) + 1)
])

fitur = [
    "jam_belajar_per_minggu",
    "belajar_kelompok_per_bulan",
    "kunjungan_perpustakaan_per_bulan",
    "jam_belajar_online_per_minggu",
]

for kolom in fitur:
    indeks = np.random.choice(df.index, size=3, replace=False)
    df.loc[indeks, kolom] = np.nan

print("1. DATA MENTAH")
print(df.head())
print("\nJumlah nilai kosong per kolom:")
print(df[fitur].isna().sum())

df_bersih = df.copy()

for kolom in fitur:
    df_bersih[kolom] = df_bersih[kolom].fillna(
        df_bersih[kolom].median()
    )

for kolom in fitur:
    df_bersih[kolom] = df_bersih[kolom].clip(lower=0)

X = df_bersih[fitur].copy()

print("\n2. HASIL PEMBERSIHAN ")
print("Jumlah nilai kosong setelah cleaning:")
print(X.isna().sum())
print("\nStatistik data:")
print(X.describe().round(2))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n3. EVALUASI JUMLAH CLUSTER")

hasil_silhouette = {}

for k in range(2, 7):
    model_uji = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    label_uji = model_uji.fit_predict(X_scaled)

    skor = silhouette_score(X_scaled, label_uji)
    hasil_silhouette[k] = skor

    print(f"k = {k}: Silhouette Score = {skor:.3f}")

k_terbaik = max(hasil_silhouette, key=hasil_silhouette.get)
print(f"\nJumlah cluster terpilih: {k_terbaik}")

kmeans = KMeans(
    n_clusters=k_terbaik,
    random_state=42,
    n_init=10
)

df_bersih["cluster"] = kmeans.fit_predict(X_scaled)

print("\n4. JUMLAH MAHASISWA PER CLUSTER")
print(df_bersih["cluster"].value_counts().sort_index())

print("\n5. PROFIL RATA-RATA SETIAP CLUSTER")
profil_cluster = (
    df_bersih.groupby("cluster")[fitur]
    .mean()
    .round(2)
)
print(profil_cluster)

print("\n6. CONTOH HASIL CLUSTERING")
print(df_bersih[
    ["id_mahasiswa"] + fitur + ["cluster"]
].head(15).to_string(index=False))

detektor = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

hasil_anomali = detektor.fit_predict(X_scaled)

df_bersih["status_pola"] = np.where(
    hasil_anomali == -1,
    "Tidak biasa",
    "Dalam pola umum"
)

print("\n7. DETEKSI POLA TIDAK BIASA")
print(df_bersih["status_pola"].value_counts())

print("\nContoh observasi yang ditandai tidak biasa:")
kolom_tampil = ["id_mahasiswa"] + fitur + ["cluster", "status_pola"]

print(
    df_bersih.loc[
        df_bersih["status_pola"] == "Tidak biasa",
        kolom_tampil
    ].head(10).to_string(index=False)
)

pca = PCA(n_components=2)
hasil_pca = pca.fit_transform(X_scaled)

df_bersih["PCA1"] = hasil_pca[:, 0]
df_bersih["PCA2"] = hasil_pca[:, 1]

print("\n8. INFORMASI PCA")
print(
    "Total variasi yang dijelaskan dua komponen:",
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
)

plt.figure(figsize=(10, 6))

for cluster_id in sorted(df_bersih["cluster"].unique()):
    subset = df_bersih[df_bersih["cluster"] == cluster_id]

    plt.scatter(
        subset["PCA1"],
        subset["PCA2"],
        label=f"Cluster {cluster_id}",
        s=65,
        alpha=0.75
    )

anomali = df_bersih[df_bersih["status_pola"] == "Tidak biasa"]

plt.scatter(
    anomali["PCA1"],
    anomali["PCA2"],
    marker="x",
    s=130,
    label="Pola tidak biasa"
)

plt.xlabel("Komponen Utama 1 (PCA1)")
plt.ylabel("Komponen Utama 2 (PCA2)")
plt.title("Pola Belajar Mahasiswa: Clustering dan Anomali")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("visualisasi_pola_belajar.png", dpi=200)
plt.show()

mahasiswa_baru = pd.DataFrame([{
    "jam_belajar_per_minggu": 5,
    "belajar_kelompok_per_bulan": 2,
    "kunjungan_perpustakaan_per_bulan": 1,
    "jam_belajar_online_per_minggu": 9,
}])

mahasiswa_baru_scaled = scaler.transform(mahasiswa_baru[fitur])

prediksi_cluster = kmeans.predict(mahasiswa_baru_scaled)[0]
prediksi_anomali = detektor.predict(mahasiswa_baru_scaled)[0]

print("\n9. ANALISIS MAHASISWA BARU")
print(mahasiswa_baru.to_string(index=False))
print(f"Cluster mahasiswa baru: {prediksi_cluster}")
print(
    "Status pola:",
    "Tidak biasa" if prediksi_anomali == -1
    else "Dalam pola umum"
)

print("\nProgram selesai.")
