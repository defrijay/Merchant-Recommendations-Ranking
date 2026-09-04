# Rekomendasi & Perankingan Merchant
### Recommender System + Learning to Rank

![Cover](assets/cover.jpg)

> Proyek ini membangun sistem rekomendasi merchant dua tahap: candidate generation (mempersempit ribuan merchant jadi 100 kandidat relevan per user) lalu ranking (mengurutkan 100 kandidat itu berdasarkan kemungkinan user benar-benar bertransaksi). Tahap 1 pakai ALS dari histori rating user-merchant, tahap 2 pakai LightGBM Ranker yang menggabungkan skor ALS dengan sinyal lain seperti popularitas, rating, dan kedekatan kategori. Semua angka di README ini dihitung di validation set yang di-split di level user (0% overlap dengan data training), jadi merepresentasikan performa model pada user yang belum pernah dilihat saat training. Hasil utama: model menaruh minimal 1 merchant relevan di Top-5 untuk **83,2% user** (vs 6,2% dengan urutan popularitas semata), setara peningkatan **~3.470% di Recall@5**.

## Daftar Isi

1. [Latar Belakang & Business Questions](#1-latar-belakang--business-questions)
2. [Sumber Data](#2-sumber-data)
3. [Metodologi](#3-metodologi)
4. [Tech Stack](#4-tech-stack)
5. [Cara Kerja Tiap Algoritma](#5-cara-kerja-tiap-algoritma)
6. [Cara Mengukur Hasilnya](#6-cara-mengukur-hasilnya)
7. [Hasil & Insight Kunci](#7-hasil--insight-kunci)
   - [7.1 BQ1: Merchant paling relevan per user](#71-bq1--merchant-paling-relevan-per-user)
   - [7.2 BQ2: Pengaruh urutan terhadap kemungkinan transaksi](#72-bq2--pengaruh-urutan-terhadap-kemungkinan-transaksi)
   - [7.3 BQ3: Fitur paling berpengaruh](#73-bq3--fitur-paling-berpengaruh)
   - [7.4 BQ4: Peningkatan relevansi vs baseline](#74-bq4--peningkatan-relevansi-vs-baseline)
8. [Rekomendasi](#8-rekomendasi)
9. [Batasan & Keterbatasan Proyek](#9-batasan--keterbatasan-proyek)
10. [Cara Menjalankan](#10-cara-menjalankan)
11. [Struktur Proyek](#11-struktur-proyek)
12. [Pengembangan Lanjutan](#12-pengembangan-lanjutan)
13. [Referensi](#13-referensi)

## 1. Latar Belakang & Business Questions

Sistem rekomendasi skala industri jarang mengandalkan satu model tunggal. Pola yang umum dipakai adalah pipeline dua tahap:

1. **Candidate generation** : mempersempit jutaan merchant jadi puluhan/ratusan kandidat yang relevan bagi seorang user.
2. **Ranking** : mengurutkan kandidat itu berdasarkan kemungkinan user benar-benar bertransaksi.

Pola dua tahap ini konsisten dengan yang dipakai platform skala besar seperti Netflix, Pinterest, dan Amazon, dan sudah mulai dikaji sifat teoretisnya dari sisi konvergensi terhadap sistem rekomendasi optimal (Jaiswal, 2024). Proyek ini membangun kedua tahap tersebut memakai data transaksi marketplace nyata (Yelp Dataset), lalu diadaptasi ke konteks rekomendasi merchant pada aplikasi pembayaran.

Empat pertanyaan yang dijawab:

| # | Pertanyaan |
|---|---|
| BQ1 | Merchant apa yang paling relevan untuk direkomendasikan ke tiap user berdasarkan histori transaksinya? |
| BQ2 | Bagaimana urutan hasil rekomendasi memengaruhi kemungkinan user benar-benar bertransaksi, dibanding urutan berbasis popularitas semata? |
| BQ3 | Fitur apa (rating merchant, kedekatan kategori, recency transaksi, dll) yang paling berpengaruh terhadap ranking? |
| BQ4 | Seberapa besar peningkatan relevansi model dibanding baseline sederhana? |

## 2. Sumber Data

Sumber: [Yelp Open Dataset](https://www.yelp.com/dataset/download) (`yelp_academic_dataset_business.json`, `yelp_academic_dataset_review.json`). File review berukuran besar (>5GB), jadi data dibaca per-chunk (streaming) dan langsung difilter agar hemat memori.

Langkah persiapan data:

1. Kota difilter otomatis berdasarkan 2 kota dengan jumlah bisnis terbanyak di dataset: Philadelphia dan Tucson.
2. Merchant dengan `review_count < 10` dibuang.
3. User dengan jumlah review `< 5` dibuang, untuk menghindari cold-start ekstrem.
4. Data displit berdasarkan waktu: transaksi sebelum 2018-01-01 untuk training, sesudahnya untuk test.

| Tahap | Jumlah |
|---|---|
| Total bisnis (raw) | 150.346 |
| Bisnis terpakai (2 kota, review_count ≥ 10) | 16.642 |
| Total baris review discan | 6.990.280 |
| Review terpakai setelah filter | 820.107 |
| Rentang tanggal review | 2005-03-02 s.d. 2022-01-19 |
| User train+test (setelah filter aktivitas) | 23.096 |
| Item unik | 16.461 |
| Baris data ranking (positive + negative sampling) | 4.771.946 |

## 3. Metodologi

**Dalam bahasa sederhana:** proyek ini bekerja dua tahap, mirip proses rekrutmen karyawan. Tahap pertama menyaring ribuan merchant jadi 100 kandidat yang kira-kira cocok buat tiap user seperti menyaring ribuan lamaran kerja jadi 100 kandidat yang layak dipanggil wawancara. Tahap kedua mengurutkan 100 kandidat itu dari yang paling mungkin sampai yang paling kecil kemungkinannya untuk benar-benar ditransaksikan user seperti wawancara final yang menentukan urutan siapa yang paling layak diterima duluan.

**Tahap 1 : Candidate Generation (ALS).** Matriks interaksi user × merchant (rating) dibangun dari `train_df`, dilatih dengan `implicit.AlternatingLeastSquares` (factors=50, iterations=20). Tiap user mendapat 100 kandidat teratas dari ALS.

**Tahap 2 : Ranking (LightGBM LambdaRank).** Kandidat dari ALS digabung dengan item populer dan sampel random sebagai negatif, lalu diberi 9 fitur. Pemilihan GBDT (LightGBM) sebagai ranker di tahap ini konsisten dengan temuan bahwa model berbasis pohon (GBDT/Random Forest) tetap unggul dibanding deep learning pada data tabular berukuran menengah seperti data ranking di proyek ini (Grinsztajn et al., 2022):

| Fitur | Deskripsi |
|---|---|
| `item_popularity` | Jumlah review merchant |
| `item_avg_rating` | Rating rata-rata merchant |
| `user_avg_rating` | Rating rata-rata yang diberikan user |
| `user_review_count` | Jumlah review yang pernah dibuat user |
| `inter_rating` | Rating user ke merchant tsb, kalau pernah berinteraksi |
| `inter_recency` | Jarak hari sejak interaksi terakhir |
| `cat_similarity` | Kemiripan kategori (cosine similarity TF-IDF) antara histori user dan merchant |
| `als_score` | Skor dot-product dari model ALS |
| `in_als_topk` | Apakah merchant masuk top-100 rekomendasi ALS |

**Dalam bahasa sederhana, 9 fitur di atas itu jawaban atas pertanyaan:** "seberapa populer merchant ini", "seberapa bagus ratingnya", "seberapa mirip kategorinya dengan yang biasa dibeli user ini", dan "seberapa dekat merchant ini dengan preferensi tersembunyi user menurut Tahap 1". Model belajar bobot tiap faktor ini dari data histori, bukan diatur manual.

**Validasi.** `df_rank` displit di level user (bukan baris) 80/20 menjadi `df_rank_train` (18.476 user, 3.816.943 baris) dan `df_rank_val` (4.620 user, 955.003 baris), dengan 0 user overlap antara keduanya. Model dilatih dengan `objective='lambdarank'`, early stopping berdasarkan `ndcg@10` di validation set. Seluruh metrik BQ1–BQ4 dihitung di `df_rank_val` : data yang sama sekali tidak dilihat model saat training.

**Kenapa ini penting buat bisnis:** model diuji ke 4.620 user yang sama sekali belum pernah "dilihat" model saat belajar supaya angka performa yang dilaporkan mencerminkan kondisi nyata di user baru, bukan cuma menghafal user yang sudah dikenal.

## 4. Tech Stack

Python · Pandas · implicit (ALS) · LightGBM Ranker · scikit-learn

## 5. Cara Kerja Tiap Algoritma

**ALS (Alternating Least Squares).** Tugasnya: dari matriks rating user × merchant yang sangat jarang terisi (sparse), cari pola tersembunyi yang menjelaskan kenapa user menyukai merchant tertentu.

**Dalam bahasa sederhana:** bayangkan tiap user dan tiap merchant punya "sidik jari selera" 50 angka rahasia yang menangkap kenapa user A cenderung suka ke merchant X. Model belajar sidik jari ini dari histori transaksi, lalu mencocokkan sidik jari user dengan sidik jari merchant untuk menebak siapa yang paling klop. Bagi pembaca yang mau lihat detail matematisnya:

Caranya, matriks rating asli `R` didekomposisi jadi dua matriks lebih kecil:

```
R ≈ P × Qᵀ
```

`P` (ukuran user × faktor) merepresentasikan preferensi tiap user, `Q` (ukuran merchant × faktor) merepresentasikan karakteristik tiap merchant, keduanya dalam 50 dimensi laten (`factors=50`). Skor prediksi user `u` ke merchant `i` dihitung lewat dot product:

```
als_score(u, i) = P_u · Q_i
```

ALS melatih `P` dan `Q` dengan bergantian: bekukan `Q`, cari `P` terbaik lewat least squares, lalu bekukan `P`, cari `Q` terbaik : diulang 20 kali (`iterations=20`) sampai error mengecil. Ini yang menghasilkan 100 kandidat kasar per user di tahap 1.

**LightGBM LambdaRank.** Beda dari model klasifikasi biasa yang menilai tiap merchant sendiri-sendiri, belajar dari **pasangan** merchant per user: kalau merchant A benar-benar ditransaksikan user dan merchant B tidak, model dihukum kalau menaruh skor B lebih tinggi dari A.

**Dalam bahasa sederhana:** model ini tidak cuma dilatih untuk menebak "merchant ini bakal ditransaksikan atau tidak" satu-satu, tapi dilatih langsung untuk menebak urutan yang benar seperti melatih juri lomba untuk menyusun ranking peserta dari terbaik ke terburuk, bukan sekadar menilai satu peserta lulus atau tidak lulus. Bobot hukumannya dibuat lebih besar untuk kesalahan di posisi atas ranking (dekat rank 1) dibanding di posisi bawah, karena posisi atas yang paling menentukan pengalaman user (merchant yang muncul di 5 teratas jauh lebih sering dilihat daripada yang di urutan ke-50). Efeknya, model dioptimalkan langsung untuk urutan, bukan cuma skor per merchant. Perbandingan terkini atas LambdaMART dan varian GBDT ranking lain menunjukkan pembobotan berbasis posisi seperti ini tetap jadi bagian penting dari performa metode-metode GBDT ranking terbaik saat ini (Lyzhin et al., 2022).

## 6. Cara Mengukur Hasilnya

Empat metrik ini dihitung untuk tiap user di validation set, lalu dirata-ratakan:

**Recall@K** : dari semua merchant yang benar-benar ditransaksikan user, berapa persen yang masuk ke K rekomendasi teratas.

```
Recall@K = (merchant relevan di top-K) / (total merchant relevan user itu)
```

*Artinya:* kalau user itu sebenarnya pernah bertransaksi di 4 merchant berbeda, dan 2 dari 4 itu muncul di 5 rekomendasi teratas, Recall@5 untuk user itu = 50%.

**Precision@K** : dari K rekomendasi yang ditampilkan, berapa persen yang benar-benar relevan.

```
Precision@K = (merchant relevan di top-K) / K
```

*Artinya:* dari 5 merchant yang ditampilkan ke user, berapa banyak yang memang "kena sasaran". Ini ukuran seberapa tidak sia-sia tiap slot rekomendasi yang ditampilkan.

**Hit Rate@K** : apakah minimal ada satu merchant relevan di top-K (1 kalau ada, 0 kalau tidak), dirata-rata lintas user.

```
Hit Rate@K = 1 jika (merchant relevan di top-K) > 0, selain itu 0
```

*Artinya:* metrik paling sederhana "apakah user ini melihat setidaknya satu rekomendasi yang cocok buat dia, ya atau tidak". Ini metrik yang paling gampang dijelaskan ke stakeholder non-teknis: 83,2% artinya 8 dari 10 user melihat minimal satu rekomendasi yang relevan di 5 teratas.

**NDCG@K (Normalized Discounted Cumulative Gain)** : mengukur relevansi sekaligus posisinya: merchant relevan yang muncul di rank 1 dihargai lebih tinggi daripada yang muncul di rank 10.

```
DCG@K  = Σ (relevansi_i / log2(i + 2))     untuk i = 0 ... K-1
NDCG@K = DCG@K / IDCG@K
```

*Artinya, dalam bahasa sederhana:* Recall dan Precision cuma menghitung "berapa banyak yang kena", tapi tidak peduli posisinya. NDCG peduli posisi rekomendasi yang tepat tapi nangkring di urutan pertama dihargai jauh lebih tinggi daripada yang tepat tapi terselip di urutan ke-10, karena user jarang scroll sampai bawah. `relevansi_i` bernilai 1 kalau merchant di posisi `i` benar-benar ditransaksikan user, 0 kalau tidak. `IDCG@K` adalah skor dari urutan ideal (semua merchant relevan ditaruh di posisi paling atas) jadi NDCG selalu berada di rentang 0–1, dengan 1 berarti urutan model sudah sama persis dengan urutan yang paling ideal.

## 7. Hasil & Insight Kunci

### 7.1 BQ1 : Merchant paling relevan per user

Di 4.620 user validation set, model berhasil menaruh minimal 1 merchant yang benar-benar ditransaksikan user di Top-5 rekomendasi untuk **83,2% user** (Hit Rate@5), dibanding cuma **6,2% user** kalau memakai urutan popularitas semata.

Precision@5 model = **0,3910** (dari 5 rekomendasi teratas, rata-rata 1,95 di antaranya benar-benar relevan) vs Precision@5 popularity = **0,0149**.

### 7.2 BQ2 : Pengaruh urutan terhadap kemungkinan transaksi

NDCG@5 model = **0,5761** vs NDCG@5 popularity = **0,0173** di validation set. Urutan hasil model menaruh merchant yang benar-benar ditransaksikan user jauh lebih dekat ke posisi atas (rank 1–5) dibanding urutan berbasis popularitas : user lebih cepat melihat merchant yang relevan buat mereka.

Tiga contoh user dari validation set (dipilih acak sebagai ilustrasi kualitatif, bukan dasar kesimpulan utama  kesimpulan BQ2 memakai rata-rata di atas):

| User | NDCG@5 Model | NDCG@5 Popularity | Verdict |
|---|---|---|---|
| `1Z5m2Pzw...` | 0,441 | 0,000 | Model lebih baik |
| `52nYCf9C...` | 0,000 | 0,000 | Sama : keduanya tidak menaikkan item relevan |
| `Zo3K-CTw...` | 0,830 | 0,000 | Model lebih baik |

### 7.3 BQ3 : Fitur paling berpengaruh

![Feature Importance](assets/bq3_feature_importance.png)

Tiga fitur teratas: `item_popularity` (474), `als_score` (468), `cat_similarity` (206), diikuti `item_avg_rating` (~190) dan `in_als_topk` (~70). Empat fitur lain : `inter_rating`, `user_review_count`, `user_avg_rating`, `inter_recency` : importance-nya mendekati nol. Cuma 5 dari 9 fitur yang benar-benar dipakai model.

Model paling mengandalkan karakteristik merchant (popularitas, rating) dan sinyal kolaboratif ALS (preferensi historis user), bukan fitur level-user (rata-rata rating user, jumlah review user) atau recency. Ini perlu diperhatikan kalau target berikutnya adalah memperbaiki rekomendasi untuk user yang jarang bertransaksi : fitur yang mestinya membantu kasus itu justru kurang termanfaatkan model saat ini.

### 7.4 BQ4 : Peningkatan relevansi vs baseline

| Metrik | Model (ALS+Ranker) | Popularity Baseline | ALS Score Baseline |
|---|---|---|---|
| Recall@5 | 0,4532 | 0,0127 | 0,0272 |
| NDCG@5 | 0,5761 | 0,0173 | 0,0397 |
| Recall@10 | 0,5846 | 0,0215 | 0,0438 |
| NDCG@10 | 0,5948 | 0,0199 | 0,0434 |

![Perbandingan Recall & NDCG: Model vs Baseline](assets/bq4_model_vs_baseline.png)

Di validation set, model naik **~3.470%** di Recall@5 dan **~3.229%** di NDCG@5 dibanding baseline popularitas. Angka persentase sebesar ini wajar secara matematis karena baseline-nya sangat kecil (Recall@5 popularitas cuma 0,0127)  bukan tanda ada yang salah, tapi juga berarti persentase ini belum disertai uji signifikansi statistik (lihat Batasan poin 3). Uplift besar ini masuk akal secara bisnis: baseline popularitas menyodorkan daftar merchant generik yang sama untuk semua user, padahal preferensi tiap user di dataset ini sangat spesifik dan beragam, jadi baseline itu memang sulit menebak merchant yang relevan buat individu tertentu.

ALS Score Baseline (skor mentah ALS tanpa ranker) juga jauh di bawah model penuh : tanda bahwa lapisan ranking (LightGBM) memberi nilai tambah signifikan di atas candidate generation saja.

## 8. Rekomendasi

1. Investigasi ulang fitur level-user (`user_review_count`, `user_avg_rating`, `inter_recency`) lewat rekayasa ulang seperti normalisasi atau binning, supaya lebih bisa ditangkap tree-based model.
2. Tambahkan uji signifikansi statistik (misalnya bootstrap confidence interval) atas selisih metrik model vs baseline, untuk memperkuat klaim uplift  terutama sebelum angka seperti "+3.470%" dikutip di luar konteks laporan ini.
3. Cek ulang cakupan kota: filter otomatis menghasilkan Philadelphia dan Tucson. Kalau target proyek memang untuk kota tertentu, pastikan kota itu memang tidak ada di versi dataset ini.

## 9. Batasan & Keterbatasan Proyek

1. Filter kota otomatis (top-2 berdasarkan jumlah bisnis) menghasilkan Philadelphia dan Tucson, bukan kota yang dipilih manual.
2. Cuma 5 dari 9 fitur yang benar-benar berkontribusi ke model (lihat BQ3); fitur level-user hampir tidak terpakai.
3. Belum ada uji signifikansi statistik atas selisih metrik model vs baseline  persentase uplift di BQ4 (mis. +3.470%) besar secara relatif karena baseline-nya kecil, dan belum ada interval kepercayaan di baliknya.

## 10. Cara Menjalankan

```bash
pip install pandas numpy scipy scikit-learn implicit lightgbm tqdm matplotlib seaborn --break-system-packages
```

1. Unduh dataset dari [Yelp Open Dataset](https://www.yelp.com/dataset/download) dan letakkan di folder `./data/`.
2. Buka `rekomendasi_merchant.ipynb`.
3. Jalankan seluruh cell dari atas ke bawah (Run All). Waktu proses candidate generation + feature building bisa memakan >1 jam tergantung spesifikasi mesin : pantau progress bar `tqdm` di tiap tahap.

## 11. Struktur Proyek

```
rekomendasi_merchant.ipynb   -> notebook utama: candidate generation (ALS), ranking (LightGBM LambdaRank), evaluasi BQ1-BQ4
assets/                      -> chart hasil evaluasi untuk README
data/                        -> file dataset Yelp (tidak disertakan di repo)
```

## 12. Pengembangan Lanjutan

1. Cross-check hasil dengan cutoff waktu berbeda (sensitivity check terhadap `TRAIN_CUTOFF`).
2. Tambahkan analisis segmentasi user (aktif vs jarang bertransaksi), untuk melihat apakah keunggulan model konsisten di semua segmen : terutama karena temuan BQ3 mengindikasikan model mungkin kurang optimal untuk user dengan histori tipis.

## 13. Referensi

Jaiswal, A. K. (2024). *Towards a theoretical understanding of two-stage recommender systems* (arXiv:2403.00802). arXiv. https://arxiv.org/abs/2403.00802

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems* (Vol. 35, pp. 507–520). Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

Lyzhin, I., Ustimenko, A., Gulin, A., & Prokhorenkova, L. (2022). *Which tricks are important for learning to rank?* (arXiv:2204.01500). arXiv. https://doi.org/10.48550/arXiv.2204.01500