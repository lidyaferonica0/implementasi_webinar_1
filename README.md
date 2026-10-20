# Studi Kasus Webinar 1- Analisis Pola Kebiasaan Belajar Mahasiswa Menggunakan Machine Learning

## Deskripsi Studi Kasus

Setiap mahasiswa memiliki kebiasaan belajar yang berbeda. Ada mahasiswa yang lebih sering belajar mandiri, belajar bersama teman, mengunjungi perpustakaan, atau menggunakan sumber belajar online. Perbedaan kebiasaan tersebut dapat dianalisis untuk mengetahui kelompok mahasiswa dengan pola belajar yang serupa.

Pada studi kasus ini, Machine Learning digunakan untuk mengelompokkan pola kebiasaan belajar mahasiswa dan mendeteksi data yang memiliki karakteristik berbeda dari pola umum. Implementasi dibuat menggunakan Python dengan data simulasi sebanyak 120 mahasiswa.

## Tujuan

1. Membersihkan dan mempersiapkan data sebelum dianalisis.
2. Mengelompokkan mahasiswa berdasarkan kebiasaan belajar menggunakan K-Means.
3. Menentukan jumlah cluster berdasarkan Silhouette Score.
4. Mendeteksi pola yang tidak biasa menggunakan Isolation Forest.
5. Menampilkan hasil pengelompokan dalam bentuk visualisasi menggunakan PCA.
6. Mencoba model pada data mahasiswa baru.

## Data yang Digunakan

Data dibuat secara simulasi menggunakan NumPy dan Pandas. Setiap data memiliki empat fitur utama:

| Fitur | Keterangan |
|---|---|
| `jam_belajar_per_minggu` | Jumlah jam belajar dalam satu minggu |
| `belajar_kelompok_per_bulan` | Frekuensi belajar kelompok dalam satu bulan |
| `kunjungan_perpustakaan_per_bulan` | Frekuensi kunjungan ke perpustakaan dalam satu bulan |
| `jam_belajar_online_per_minggu` | Jumlah jam belajar online dalam satu minggu |

Setiap mahasiswa juga memiliki ID untuk membedakan satu data dengan data lainnya. Beberapa nilai kosong sengaja ditambahkan untuk menguji proses pembersihan data.

## Metode yang Digunakan

### 1. Pembersihan Data

Nilai kosong pada setiap fitur diisi menggunakan median kolom masing-masing. Setelah itu, nilai fitur dibatasi agar tidak bernilai negatif.

### 2. Standardisasi Data

`StandardScaler` digunakan untuk menyamakan skala setiap fitur sebelum proses clustering dan deteksi anomali dilakukan.

### 3. Clustering dengan K-Means

K-Means digunakan untuk mengelompokkan mahasiswa berdasarkan kemiripan kebiasaan belajar. Program menguji jumlah cluster dari 2 hingga 6. Silhouette Score digunakan untuk membandingkan hasil pengelompokan, kemudian jumlah cluster dengan skor tertinggi dipilih untuk model akhir.

Hasilnya ditampilkan dalam bentuk jumlah mahasiswa dan rata-rata karakteristik setiap cluster.

### 4. Deteksi Anomali dengan Isolation Forest

Isolation Forest digunakan untuk mendeteksi data yang memiliki pola berbeda dari sebagian besar data lainnya. Parameter `contamination=0.05` digunakan untuk menetapkan perkiraan proporsi anomali sebesar 5%.

Data yang ditandai sebagai pola tidak biasa merupakan hasil deteksi model, bukan penilaian bahwa kebiasaan belajar mahasiswa tersebut buruk.

### 5. Visualisasi dengan PCA

PCA digunakan untuk mereduksi empat fitur menjadi dua komponen utama agar data dapat ditampilkan dalam grafik dua dimensi. Grafik memperlihatkan kelompok hasil clustering serta data yang ditandai sebagai pola tidak biasa.

### 6. Pengujian Mahasiswa Baru

Program menguji satu data mahasiswa baru dengan karakteristik berikut:

- Belajar mandiri selama 5 jam per minggu.
- Belajar kelompok sebanyak 2 kali per bulan.
- Mengunjungi perpustakaan 1 kali per bulan.
- Belajar online selama 9 jam per minggu.

Data tersebut diproses menggunakan scaler yang sama dengan data sebelumnya. Model kemudian menentukan cluster dan status pola mahasiswa baru.

## Library yang Digunakan

- **NumPy** untuk pembuatan dan pengolahan data numerik.
- **Pandas** untuk mengelola data dalam bentuk tabel.
- **Matplotlib** untuk membuat grafik visualisasi.
- **Scikit-learn** untuk standardisasi, clustering, evaluasi cluster, PCA, dan deteksi anomali.

## Struktur Folder

```text
webinar_1/
├── README.md
├── webinar_1.py
├── requirements.txt
└── visualisasi_pola_belajar.png
```

File `visualisasi_pola_belajar.png` dihasilkan secara otomatis ketika program dijalankan sampai selesai.

## Cara Menjalankan Program

Pastikan Python sudah terpasang di komputer. Buka terminal pada folder `webinar_1`, kemudian jalankan perintah berikut untuk memasang library yang diperlukan:

```bash
python -m pip install -r requirements.txt
```

Setelah instalasi selesai, jalankan program menggunakan perintah:

```bash
python webinar_1.py
```

Program akan menampilkan data awal, hasil pembersihan, Silhouette Score, jumlah mahasiswa per cluster, profil rata-rata setiap cluster, hasil deteksi anomali, informasi PCA, dan prediksi mahasiswa baru. Grafik hasil analisis disimpan dalam file `visualisasi_pola_belajar.png` di direktori kerja saat program dijalankan.

## Hasil dan Pembahasan

Program menghasilkan pengelompokan mahasiswa berdasarkan karakteristik kebiasaan belajar. Setiap cluster memiliki nilai rata-rata fitur yang dapat digunakan untuk memahami perbedaan pola belajar antarkelompok.

Program juga menampilkan data yang ditandai sebagai pola tidak biasa oleh Isolation Forest serta grafik yang memperlihatkan persebaran data berdasarkan dua komponen utama PCA.

Nomor cluster hanya berfungsi sebagai label kelompok, bukan sebagai urutan kemampuan atau prestasi mahasiswa. Hasilnya perlu ditafsirkan berdasarkan karakteristik data pada masing-masing cluster.

Karena data yang digunakan merupakan data simulasi, hasil analisis ini belum dapat digunakan untuk menyimpulkan kebiasaan belajar mahasiswa secara nyata.

## Kesimpulan

Studi kasus ini menerapkan beberapa metode Machine Learning untuk menganalisis pola kebiasaan belajar mahasiswa. Prosesnya meliputi pembersihan data, standardisasi fitur, pengelompokan menggunakan K-Means, evaluasi dengan Silhouette Score, deteksi anomali menggunakan Isolation Forest, serta visualisasi menggunakan PCA.

Program ini dapat dikembangkan lebih lanjut dengan menggunakan data kebiasaan belajar mahasiswa yang dikumpulkan secara nyata agar hasil analisis dapat diuji pada kondisi sebenarnya.
