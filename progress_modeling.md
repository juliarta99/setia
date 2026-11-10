# Persiapan Laporan Progress — Kelompok 1 Sains Data

**Topik:** Retensi pelanggan ritel · **Dataset:** Online Retail II (UCI) · **Notebook:** `A_Kelompok1_Data_Preparation` & `A_Kelompok1_Modeling` (model versi 2)

Isi file ini:
- §1 ringkasan
- §2 istilah
- §3 data preparation (termasuk §3.6 alasan split 75/25)
- §4 fitur baru `CustomerAge`
- §5 modeling (termasuk arti tiap segmen, alasan silhouette kecil, §5.4 alasan pemilihan metrik)
- §6 kenapa masih ada kesalahan
- §7 deployment (joblib)
- §8 pertanyaan kritis

Semua angka diambil dari output notebook.

---

## 1. Ringkasan 1 Menit

> Dari **1.067.371 baris transaksi** (Des 2009 – Des 2011), setelah cleaning 7 tahap tersisa **808.582 baris** milik **5.875 pelanggan**. Kami membangun 3 model untuk membantu toko mempertahankan pelanggan:
> 1. **Churn (classification)**: siapa yang tidak akan belanja 90 hari ke depan. **Random Forest**, ROC-AUC **0,811**.
> 2. **Segmentasi (clustering)**: 5 tipe pelanggan dan aksinya. **K-Means k = 5**, termasuk segmen **BARU**.
> 3. **Rekomendasi (association rules)**: produk yang sering dibeli bersama. **FP-Growth**, Hit@5 **11,8 %** (4,7× lebih baik daripada produk terlaris).
>
> Semuanya sudah terpasang di web Laravel dan diperbarui otomatis saat ada transaksi baru.

**Pembagian peran 3 model:** churn menjawab **siapa** yang perlu ditangani, segmen menjawab **aksi apa** yang cocok, rekomendasi menjawab **produk apa** yang ditawarkan.

---

## 2. Istilah Penting

### 2.1 Data & fitur
| Istilah | Arti | Kenapa penting di proyek ini |
|---|---|---|
| **Churn** | Pelanggan berhenti belanja. Di sini: **tidak belanja dalam 90 hari** setelah cutoff | Target yang diprediksi |
| **RFM** | **R**ecency (hari sejak belanja terakhir), **F**requency (jumlah invoice), **M**onetary (total belanja £) | Fitur standar analisis pelanggan |
| **TenureDays** | Hari antara pembelian **pertama** dan **terakhir** | Lama hubungan aktif pelanggan |
| **CustomerAge** | Hari sejak pembelian **pertama** sampai tanggal acuan = Recency + TenureDays | Membedakan pelanggan baru vs lama (lihat §4) |
| **RecencyRatio** | Recency ÷ (rata-rata jarak antar-pembelian + 1) | Lama tidak belanja dibanding kebiasaannya sendiri |
| **AvgBasketValue** | Rata-rata nilai per baris belanja | Pembeli barang murah vs mahal |
| **UniqueProducts** | Jumlah produk berbeda yang pernah dibeli | Keragaman belanja |
| **ReturnRate** | Invoice batal ÷ total invoice | Kebiasaan retur |
| **Cutoff / tanggal acuan** | Batas waktu: fitur dihitung sebelum cutoff, label sesudahnya (11 Sep 2011) | Mencegah leakage |
| **Target leakage** | Informasi label "bocor" ke fitur, sehingga skor tinggi palsu | Alasan label tidak dibuat dari `Recency > 90` |
| **Observation / performance window** | Periode pengamatan fitur / periode penentuan label | Nama ilmiah teknik label temporal kami |
| **Outlier, IQR, capping (winsorize)** | Nilai ekstrem; IQR = Q3 − Q1; capping = memangkas nilai ke batas, bukan menghapus baris | Pembelian grosir nyata tidak dibuang |
| **log1p** | log(1 + x) | Membuat distribusi miring jadi lebih simetris, aman untuk nilai 0 |
| **StandardScaler / z-score** | (x − rata-rata) ÷ standar deviasi, sehingga mean 0 dan std 1 | Fitur satuan berbeda bisa dibandingkan (wajib untuk K-Means & LR) |
| **One-hot / basket** | Matriks invoice × produk berisi True/False | Format input FP-Growth |

### 2.2 Classification (churn)
| Istilah | Arti |
|---|---|
| **Train / test split** | Data latih (75 %) dan data uji (25 %) yang dipisah sejak awal |
| **Stratified** | Proporsi churn (56,6 %) dijaga sama di train dan test |
| **Cross-validation (5-fold)** | Data train dibagi 5; bergantian 4 bagian untuk latih dan 1 untuk validasi; skornya dirata-rata |
| **GridSearchCV** | Mencoba semua kombinasi hyperparameter dengan CV, lalu memilih yang terbaik |
| **Hyperparameter** | Pengaturan model sebelum latih, misalnya `max_depth` |
| **Pipeline** | Rangkaian langkah (log1p → scaler → model) yang dilatih dan disimpan sebagai satu objek |
| **Baseline (DummyClassifier)** | Model "bodoh" yang selalu menebak kelas mayoritas. Model asli harus jauh di atasnya |
| **Probabilitas churn** | Output model 0–1 |
| **Threshold** | Batas probabilitas untuk keputusan "churn". Kami memakai **0,575** |
| **Youden's J** | Metode memilih threshold: maksimum (TPR − FPR) |
| **Confusion matrix** | Tabel benar/salah: TN, FP (alarm palsu), FN (terlewat), TP |
| **Accuracy** | Porsi tebakan benar |
| **Precision** | Dari yang ditandai churn, berapa yang benar churn |
| **Recall (TPR)** | Dari yang benar churn, berapa yang tertangkap |
| **F1** | Rata-rata harmonis precision dan recall |
| **ROC-AUC** | Peluang model memberi skor lebih tinggi ke pelanggan churn daripada yang tidak. 0,5 = acak, 1 = sempurna |
| **Permutation importance** | Seberapa turun ROC-AUC bila satu fitur diacak, artinya seberapa penting fitur itu |
| **Overfitting** | Model hafal data latih tetapi buruk di data baru |
| **Uji berpasangan (paired CV)** | Dua versi fitur dibandingkan pada fold yang **sama persis** (20 fold) |

### 2.3 Clustering (segmentasi)
| Istilah | Arti | Arah baik |
|---|---|---|
| **K-Means** | Mengelompokkan data ke k pusat (centroid) terdekat | – |
| **Centroid** | Titik pusat (rata-rata) sebuah cluster | – |
| **Inertia (WCSS)** | Jumlah kuadrat jarak titik ke centroid-nya. Selalu turun bila k bertambah, jadi hanya untuk **elbow** | ↓ |
| **Elbow** | Titik "siku" grafik inertia vs k | – |
| **Silhouette** | −1..1: seberapa dekat titik ke cluster-nya dibanding ke cluster lain | ↑ |
| **Davies-Bouldin (DBI)** | Rasio (sebaran A + sebaran B) ÷ jarak pusat A–B, dirata-rata | ↓ (0 ideal) |
| **ARI (Adjusted Rand Index)** | Kemiripan dua hasil clustering, 1 = identik. Dipakai untuk uji stabilitas antar random seed | ↑ |
| **Z-score rata-rata per segmen** | (rata-rata segmen − rata-rata semua) ÷ std. +1 = 1 std di atas rata-rata | Untuk heatmap profil |

### 2.4 Association rules (rekomendasi)
| Istilah | Arti |
|---|---|
| **Itemset** | Kumpulan produk dalam satu invoice |
| **Antecedent → Consequent** | "Jika membeli A → kemungkinan membeli B" |
| **Support** | Porsi invoice yang memuat A dan B |
| **Confidence** | P(B \| A): dari yang membeli A, berapa persen juga membeli B |
| **Lift** | Confidence ÷ P(B). > 1 berarti lebih sering dari kebetulan; kami memakai **≥ 2** |
| **FP-Growth** | Algoritma pencari itemset sering, lebih cepat dari Apriori dengan hasil sama |
| **Hit Rate@5** | Porsi kasus di mana produk yang benar ada di 5 rekomendasi teratas |

---

## 3. Data Preparation

### 3.1 Cleaning 7 tahap
| Tahap | Dibuang | Alasan |
|---|---|---|
| 1. Standardisasi tipe & spasi | 0 | Teks konsisten (`"ABC "` = `"ABC"`) |
| 2. Hapus duplikat | 12.133 | Baris identik di semua kolom = kesalahan input |
| 3. Tandai retur (`IsCancelled`) | 0 | Invoice berawalan `C`. **Tidak dibuang**, dipakai untuk `ReturnRate` |
| 4. Hapus Customer ID kosong | **242.870** | Tidak bisa dihubungkan ke pelanggan. ID tidak bisa diimputasi |
| 5. Hapus kode non-produk | 3.725 | `POST` ongkir, `M` manual, `D` diskon, `BANK CHARGES`: biaya, bukan barang |
| 6. Hapus Price ≤ 0 | 61 | Bukan penjualan riil |
| 7. Capping IQR × 3 | 0 (dipangkas) | Quantity ≤ 42, Price ≤ 11,25. 37.269 dan 18.147 nilai dipangkas |

Hasil: **808.582 baris** (75,8 %), 43.876 invoice, 5.875 pelanggan, 4.634 produk, 41 negara.

### 3.2 Dari 1 juta baris ke 5.260 pelanggan
1 baris mentah = **1 produk dalam 1 invoice**, bukan 1 pelanggan.
```
1.067.371 baris ─cleaning─▶ 808.582 baris = 5.875 pelanggan
  ├─ − 23 (hanya punya retur)                      = 5.852 → clustering
  │   − 592 (pelanggan baru setelah cutoff, belum
  │          punya riwayat untuk fitur)            = 5.260 → classification
  └─ per invoice: 43.876 − retur − invoice berisi produk langka = 36.280 → association
```

### 3.3 Label churn tanpa leakage
- Fitur dihitung dari transaksi **sebelum 11 Sep 2011 12:50** (OUTPUT 2).
- `Churn = 1` bila pelanggan **tidak membeli** dalam 90 hari sesudahnya.
- **Kenapa tidak `Recency > 90`?** Label akan identik dengan fitur (leakage). Bukti datanya tidak sama: **596** pelanggan dengan Recency ≤ 90 tetap churn, dan **969** dengan Recency > 90 ternyata kembali.
- **Kenapa 90 hari?** Grafik CDF jarak antar-pembelian menunjukkan **83,2 %** pembelian ulang terjadi ≤ 90 hari (median 25 hari).

### 3.4 Lima output
| File | 1 baris = | Dipakai untuk | Contoh isi |
|---|---|---|---|
| `01_transaksi_clean.csv` (808.582 × 11) | 1 produk dalam invoice | sumber semua dataset | `489434, 85048, 15CM CHRISTMAS GLASS BALL 20 LIGHTS, 12, 2009-12-01 07:45, 6.95, 13085, United Kingdom, …, IsCancelled=False, TotalPrice=83.4` |
| `02_transaksi_clean_tanpa_90_hari_terakhir.csv` (647.855 × 11) | sama | fitur churn | kolom sama, berhenti di 11 Sep 2011 12:46 |
| `03_classification_ready.csv` (5.260 × 8) | 1 pelanggan | model churn | lihat tabel di bawah |
| `03_..._scaled.csv` | 1 pelanggan | model linear/jarak | nilai log1p + z-score (mean 0, std 1) |
| `04_clustering_ready_scaled.csv` (5.852 × 7) | 1 pelanggan | K-Means | 6 fitur ter-scale, mis. #12346 Recency 0,86 |
| `04_clustering_profil.csv` (5.852 × 10) | 1 pelanggan | profil segmen | `12346, R=326, F=3, M=213.04, Avg=8.52, Uniq=25, CustomerAge=647, Tenure=321, Return=0.25` |
| `05_association_basket_onehot.pkl` (36.280 × 2.765) | 1 invoice | FP-Growth | True/False per produk (kepadatan 0,72 %) |
| `05_association_transaksi.csv` | invoice–produk | format panjang | `489434, PINK CHERRY LIGHTS` |
| `05_association_daftar_produk.csv` | 1 produk | daftar & support | `WHITE HANGING HEART T-LIGHT HOLDER, 4888 invoice, support 0.135` |

**Contoh `03_classification_ready.csv`** (fitur per cutoff):
| Customer ID | Recency | Frequency | Monetary | CustomerAge | RecencyRatio | Churn | Split |
|---|---|---|---|---|---|---|---|
| 12346 | 236 | 3 | 213,04 | 557 | 1,46 | 1 | train |
| 12347 | 40 | 6 | 3.890,26 | 314 | 0,72 | 0 | train |
| 12350 | 220 | 1 | 294,40 | 220 | 1,00 | 1 | train |

Tipe data: `Customer ID`, `Recency`, `Frequency`, `CustomerAge`, `Churn` = **int**; `Monetary`, `RecencyRatio` = **float**; `Split` = **string** (`train`/`test`).

**Kenapa Recency #12346 = 236 di file 03 tetapi 326 di file 04?** Acuannya berbeda: file 03 dihitung sampai cutoff (anti-leakage), file 04 sampai akhir data.

### 3.5 Pertanyaan data preparation
- **Kenapa capping, bukan hapus outlier?** Nilai besar di ritel sering nyata (pembeli grosir). × 3 dipakai karena data sangat miring; × 1,5 akan memangkas terlalu banyak transaksi wajar.
- **Kenapa produk < 50 kemunculan dibuang?** Support terlalu kecil, sehingga rule hanya kebetulan. 47 % produk dibuang, tetapi 94,6 % volume transaksi tetap ada.
- **Kenapa scaler di-fit hanya di train?** Kalau di-fit di seluruh data, statistik data test ikut "bocor" ke model.

### 3.6 Kenapa split 75 % train / 25 % test?
Data churn berisi 5.260 pelanggan, sehingga train = **3.945** dan test = **1.315**.

| Pertimbangan | Penjelasan |
|---|---|
| **Test harus cukup besar agar skornya bisa dipercaya** | Dengan 1.315 pelanggan (744 churn, 571 tidak), ROC-AUC test punya interval kepercayaan 95 % **0,785–0,834** (bootstrap). Kalau test terlalu kecil (mis. 10 % ≈ 526), intervalnya melebar dan skor bisa "beruntung" atau "sial". |
| **Train harus cukup untuk belajar dan CV** | 3.945 pelanggan dibagi 5-fold, sehingga tiap fold validasi ± 790 pelanggan dan fold latih ± 3.150. Masih cukup untuk Random Forest dan GridSearch. |
| **Sudah diuji empiris** | 10× split acak: 80/20 → AUC berfluktuasi ± **0,009**; **75/25 → ± 0,007**; 70/30 → ± 0,007. Rata-ratanya sama. 80/20 lebih fluktuatif karena test kecil; 70/30 tidak lebih stabil, tetapi membuang data latih. |
| **Konsisten dengan tugas sebelumnya** | Sama dengan tugas Seleksi Fitur (TA IV) dan default `train_test_split` scikit-learn (`test_size = 0.25`), sehingga hasil antar-tugas bisa dibandingkan. |
| **Stratified** | `stratify = Churn` menjaga proporsi churn **56,6 %** sama di train dan test. Tanpa stratify, test bisa kebetulan berisi lebih banyak/sedikit churn sehingga skor bias. |
| **`random_state = 42`** | Pembagian bisa diulang persis oleh siapa pun (reproducible). Angka 42 hanyalah konvensi; nilai lain juga sah, asal tetap. |

**Kenapa masih perlu cross-validation kalau sudah ada test?** Test hanya boleh dipakai **sekali** di akhir. Pemilihan model, hyperparameter, dan threshold dilakukan dengan 5-fold CV **di dalam train**. Kalau semua itu dipilih dengan melihat test, skor test tidak lagi jujur (overfit ke test).

**Kenapa acak per pelanggan, bukan berdasarkan waktu?** Unsur waktu sudah ditangani **label temporal**: semua pelanggan memakai cutoff yang sama, fitur sebelum dan label sesudah. Split acak memastikan model diuji pada **pelanggan yang belum pernah dilihat**. Pengembangan berikutnya: *out-of-time validation* (latih dengan cutoff lebih awal, uji dengan cutoff berikutnya).

**Kenapa clustering dan association tidak memakai split 75/25?**
- **Clustering** tidak punya label, jadi tidak ada "jawaban benar" untuk diuji. Evaluasinya memakai metrik internal dan stabilitas.
- **Association rules** diuji dengan **split waktu**: rules dari data sebelum 90 hari terakhir, lalu diuji pada invoice 90 hari terakhir. Ini lebih realistis karena rekomendasi dipakai untuk transaksi masa depan.

---

## 4. Fitur Baru `CustomerAge` & `RecencyRatio`

### 4.1 Masalah yang diperbaiki
Di versi 1 (RFM saja), pelanggan yang **baru belanja sekali 2 hari lalu** punya Frequency 1, Monetary kecil, dan sedikit produk. Itu **mirip** pembeli sekali yang sudah lama pergi. Akibatnya **84 dari 190 pelanggan baru masuk segmen HILANG** dan 52 masuk BERISIKO. Model tidak tahu **kapan pelanggan mulai belanja**.

### 4.2 Cara mendapatkan
Dari riwayat transaksi tiap pelanggan (sebelum tanggal acuan `ref`):
```
first     = tanggal pembelian pertama        last = tanggal pembelian terakhir
Recency     = (ref − last).days
TenureDays  = (last − first).days
CustomerAge = (ref − first).days  =  Recency + TenureDays

avg_interval = TenureDays / (Frequency − 1)   jika Frequency > 1
             = CustomerAge                    jika Frequency = 1 (belum punya ritme)
RecencyRatio = Recency / (avg_interval + 1)
```
Di kode: `add_derived()` (notebook dan `ml_service/features.py`, isinya identik).

### 4.3 Contoh perhitungan (dari `03_classification_ready.csv`)
| Pelanggan | Recency | TenureDays | Freq | CustomerAge | avg_interval | RecencyRatio | Makna |
|---|---|---|---|---|---|---|---|
| #12346 | 236 | 321 | 3 | 236 + 321 = **557** | 321 ÷ 2 = 160,5 | 236 ÷ 161,5 = **1,46** | sudah 1,5× lebih lama dari ritmenya → tanda bahaya |
| #12347 | 40 | 274 | 6 | **314** | 274 ÷ 5 = 54,8 | 40 ÷ 55,8 = **0,72** | masih dalam ritme normal |
| #12350 | 220 | 0 | 1 | **220** | = 220 | 220 ÷ 221 = **1,00** | sekali beli, 220 hari lalu |
| #13436 (contoh pelanggan baru) | 2 | 0 | 1 | **2** | 2 | 0,67 | baru datang → segmen **BARU** |

### 4.4 Pertanyaan
- **Kenapa tidak `TenureDays` langsung?** Sudah diuji, hasilnya justru lebih buruk: pelanggan baru yang masuk HILANG naik **84 → 132**. `TenureDays` = 0 untuk semua pembeli sekali, baik yang datang kemarin maupun setahun lalu. `CustomerAge` tetap memuat `TenureDays`, tetapi ditambah Recency sehingga keduanya terbedakan (2 hari vs 400 hari).
- **Kenapa `RecencyRatio` hanya di churn, tidak di clustering?** Distribusinya sangat miring (skewness 9,9) dan informasinya tumpang tindih dengan Recency. Untuk K-Means yang berbasis jarak, fitur seperti itu menambah bobot ganda ke Recency. Clustering cukup memakai `CustomerAge`.
- **Buktinya lebih baik?**

| | Versi 1 | Versi 2 |
|---|---|---|
| Churn (20 fold berpasangan) | ROC-AUC 0,7946 | **0,8000**, lebih baik di **20/20 fold** |
| Churn (test) | 0,805 | **0,811** |
| Pelanggan baru (190) salah masuk HILANG/BERISIKO | 136 | **0** (semua masuk BARU) |
| Davies-Bouldin / ARI | 1,256 / 0,988 | **1,187 / 0,997** |

---

## 5. Modeling

### 5.1 Model 1: Churn (classification)

**Kandidat yang diuji dan alasannya:**
| Model | Kenapa diuji | Hasil CV ROC-AUC |
|---|---|---|
| Dummy (baseline) | Patokan minimum: model asli wajib jauh di atasnya | 0,500 |
| Logistic Regression | Model linear sederhana, mudah diinterpretasi. Mewakili keluarga **linear** | 0,788 |
| **Random Forest** | Ensemble **bagging** pohon keputusan. Menangkap pola non-linear, tahan outlier | **0,800** ✅ |
| HistGradientBoosting | Ensemble **boosting**, umumnya terkuat untuk data tabular | 0,799 |

Ketiganya mewakili 3 pendekatan berbeda: linear, bagging, dan boosting. Kalau ketiganya memberi hasil mirip (0,79–0,80), berarti batas kemampuan ada di **informasi fiturnya**, bukan di pilihan algoritma.

**Kenapa tidak SVM / KNN / Neural Network / XGBoost?**
- **SVM:** probabilitasnya tidak alami (butuh kalibrasi tambahan) dan lambat saat tuning.
- **KNN:** sangat sensitif skala dan outlier, dan prediksinya lambat.
- **Neural Network:** data hanya 3.945 baris dan 5 fitur; rawan overfit dan sulit dijelaskan.
- **XGBoost:** keluarga yang sama dengan HistGradientBoosting (boosting); sudah terwakili tanpa library tambahan.

**Kenapa Random Forest terbaik?**
1. **Pola non-linear dan interaksi.** Contoh: Recency 60 hari berbahaya bagi pelanggan yang sering belanja, tetapi normal bagi yang jarang. Pohon keputusan menangkap interaksi seperti ini; Logistic Regression hanya menarik garis lurus, sehingga skornya paling rendah (0,788).
2. **Bagging menurunkan variance.** 300 pohon dilatih pada sampel acak berbeda, lalu hasilnya dirata-rata. Kesalahan pohon individual saling menutupi, sehingga stabil untuk data berukuran sedang dan noisy.
3. **Tahan outlier dan distribusi miring.** Pohon membagi data berdasarkan urutan nilai, bukan besar nilainya, jadi pembeli grosir tidak merusak model.
4. **Dibatasi agar tidak overfit.** `max_depth = 6` dan `min_samples_leaf = 5` dipilih lewat GridSearch.
5. **Jujurnya:** selisih RF (0,8002) dan HistGradientBoosting (0,7991) hanya 0,001, masih dalam variasi antar-fold. Keduanya setara. RF dipilih **otomatis** oleh notebook karena CV ROC-AUC tertinggi, bukan dipilih manual. Boosting biasanya unggul pada data besar dengan banyak fitur; di sini datanya kecil dan fiturnya hanya 5, jadi keunggulannya tidak terlihat.

**Hasil di data test (1.315 pelanggan):**
| Metrik | Nilai |
|---|---|
| ROC-AUC | **0,811** (IK 95 % 0,785–0,834) |
| Accuracy | 74,1 % |
| Precision / Recall / F1 (churn) | 0,777 / 0,759 / 0,768 |
| Threshold (Youden) | 0,575 |

| | Diprediksi tidak churn | Diprediksi churn |
|---|---|---|
| Aktual tidak churn (571) | 409 benar | 162 alarm palsu |
| Aktual churn (744) | 179 terlewat | 565 tertangkap |

**Level risiko untuk web:**
| Level | Probabilitas | Jumlah | Churn aktual |
|---|---|---|---|
| RENDAH | < 0,4 | 354 | 20,6 % |
| SEDANG | 0,4–0,7 | 471 | 55,4 % |
| TINGGI | ≥ 0,7 | 490 | 83,7 % |

**Pertanyaan:**
- **Kenapa ROC-AUC sebagai metrik utama, bukan accuracy?** Web memakai **probabilitas** untuk mengurutkan prioritas, dan ROC-AUC menilai kualitas urutan itu. Accuracy menipu: Dummy yang selalu menebak "churn" sudah 56,6 % (F1 0,72), padahal ROC-AUC-nya 0,50.
- **Apa itu threshold, dan kenapa 0,575?** Batas probabilitas untuk memutuskan "churn". Dipilih dengan Youden's J dari prediksi *out-of-fold* data **train** (bukan test, agar tidak curang), sehingga recall dan alarm palsu seimbang.
- **Data tidak seimbang?** 56,6 : 43,4 hanya sedikit tidak seimbang. Penanganannya: stratified split dan CV, ROC-AUC, dan threshold Youden. SMOTE tidak dipakai karena merusak kalibrasi probabilitas (yang dipakai untuk level risiko) dan membuat pelanggan sintetis.
- **Overfitting?** Test dipisah sejak awal dan hanya dipakai sekali. CV 0,800 vs test 0,811 berdekatan, dan kedalaman pohon dibatasi.
- **Fitur paling berpengaruh?** Permutation importance: **Recency 0,113**, Monetary 0,034, Frequency 0,019, CustomerAge 0,006, RecencyRatio 0,006. Dua fitur baru terlihat kecil karena tumpang tindih dengan Recency; bukti manfaatnya ada di uji berpasangan 20/20 fold.
- **Kenapa model final di-refit ke seluruh data?** Setelah evaluasi selesai, parameter sudah dikunci, jadi melatih ulang pada train + test memakai informasi maksimal. Skor yang dilaporkan tetap dari test sebelum refit.
- **"Saran aksi" dari mana?** Aturan bisnis per segmen (praktik pemasaran RFM), bukan output model.

### 5.2 Model 2: Segmentasi (K-Means)
Pipeline: `log1p` → `StandardScaler` → `KMeans`, dengan 6 fitur (Recency, Frequency, Monetary, AvgBasketValue, UniqueProducts, CustomerAge).

| k | Inertia | Silhouette ↑ | Davies-Bouldin ↓ | Cluster terkecil |
|---|---|---|---|---|
| 3 | 18.307 | **0,320** | **1,142** | 14,5 % |
| 4 | 15.576 | 0,250 | 1,247 | 13,4 % |
| **5** | 13.751 | 0,254 | 1,187 | 12,5 % |
| 6 | 12.451 | 0,254 | 1,203 | 7,6 % |

**Profil segmen (median):**
| Segmen | % | Recency | Freq | Monetary | CustomerAge | Aksi |
|---|---|---|---|---|---|---|
| LOYAL | 20,8 | 17 | 13 | £ 4.472 | 676 | program VIP |
| POTENSIAL | 27,7 | 101 | 4 | £ 1.257 | 578 | cross-sell, bundling |
| BERISIKO | 15,2 | 306 | 2 | £ 473 | 523 | win-back voucher |
| HILANG | 23,7 | 411 | 1 | £ 248 | 470 | re-aktivasi murah |
| **BARU** | 12,5 | 29 | 1 | £ 380 | **61** | onboarding, dorong pembelian kedua |

**Arti tiap segmen** (dihitung ulang dengan model aktif pada 5.852 pelanggan):

| Segmen | Rata-rata peluang churn | Risiko TINGGI | Porsi revenue |
|---|---|---|---|
| LOYAL | **17 %** | 1 % | **70,4 %** |
| POTENSIAL | 54 % | 21 % | 19,5 % |
| BERISIKO | 72 % | 57 % | 4,0 % |
| HILANG | **84 %** | 89 % | 3,0 % |
| BARU | 44 % | 4 % | 3,0 % |

**🟢 LOYAL: Pelanggan Loyal Bernilai Tinggi**
- **Siapa:** baru saja belanja (median 17 hari lalu), sering (13×), belanja terbesar (£ 4.472), produk paling beragam (178). Sudah lama menjadi pelanggan (676 hari).
- **Arti bisnis:** hanya 20,8 % pelanggan, tetapi menyumbang **70 % revenue** (pola Pareto). Kehilangan satu pelanggan LOYAL sama dengan kehilangan puluhan pelanggan lain.
- **Aksi:** program VIP/loyalitas, akses awal produk baru. Tujuannya mempertahankan, bukan diskon besar (mereka sudah mau membayar).

**🔵 POTENSIAL: Pelanggan Potensial**
- **Siapa:** masih aktif (101 hari), sudah beberapa kali belanja (4×), nilai menengah (£ 1.257), pelanggan lama (578 hari).
- **Arti bisnis:** kelompok terbesar (27,7 %) dan "tengah": peluang churn ± 50 %. Bisa naik menjadi LOYAL atau turun menjadi BERISIKO.
- **Aksi:** cross-sell dan bundling (di sinilah association rules paling berguna) untuk menaikkan frekuensi dan nilai belanja.

**🟠 BERISIKO: Pelanggan Mulai Hilang**
- **Siapa:** **dulu** pernah aktif (rata-rata 2×, hubungan aktif ± 161 hari), tetapi sudah **± 10 bulan** tidak belanja (306 hari).
- **Bedanya dengan HILANG:** BERISIKO punya **riwayat hubungan** (tenure > 0, mayoritas belanja > 1×), jadi masih ada "ikatan" untuk diajak kembali.
- **Arti bisnis:** peluang churn 72 %. Ini **kesempatan terakhir** sebelum benar-benar hilang.
- **Aksi:** kampanye win-back: voucher, email pengingat, "kami rindu Anda".

**🔴 HILANG: Pelanggan Hilang (Sekali Beli)**
- **Siapa:** mayoritas (67 %) hanya belanja **sekali**, sudah **± 14 bulan** lalu (411 hari), nilai kecil (£ 248), tenure 0.
- **Arti bisnis:** peluang churn tertinggi (84 %) dan revenue hanya 3 %. Biaya membujuk mereka sering lebih besar dari hasilnya.
- **Aksi:** re-aktivasi **murah** (email massal/newsletter), bukan voucher mahal.

**🟣 BARU: Pelanggan Baru**
- **Siapa:** **umur pelanggan sangat muda** (median CustomerAge 61 hari, segmen lain > 470 hari), baru belanja (29 hari), belum sering (median 1×).
- **Bedanya dengan HILANG:** sama-sama Frequency kecil, tetapi BARU **baru datang**, bukan sudah pergi. Inilah yang dibedakan oleh `CustomerAge`.
- **Arti bisnis:** peluang churn 44 %, dan hampir tidak ada yang berisiko TINGGI (4 %). Momen paling penting bagi mereka adalah **pembelian kedua**: pelanggan yang belanja 2× jauh lebih mungkin menjadi pelanggan tetap.
- **Aksi:** onboarding (ucapan selamat datang, rekomendasi produk awal, insentif pembelian kedua).

**Cara cepat membedakan:**
```
Baru belanja?  ── ya ──▶ Sudah lama jadi pelanggan & sering? ── ya ──▶ LOYAL
     │                                  └── tidak, umur muda ─────────▶ BARU
     │                                  └── menengah ─────────────────▶ POTENSIAL
     └── tidak ──▶ Pernah punya riwayat (belanja > 1×)? ── ya ──▶ BERISIKO
                                                └── tidak ──────▶ HILANG
```
(Ini hanya ilustrasi; K-Means memakai jarak ke centroid di 6 fitur sekaligus, bukan aturan if-else.)

**Bukti segmen bermakna:** rata-rata peluang churn naik berurutan LOYAL 17 % → POTENSIAL 54 % → BERISIKO 72 % → HILANG 84 %, padahal **label churn tidak dipakai** saat clustering. Ini validasi eksternal bahwa segmen mencerminkan perilaku nyata.

**Pertanyaan:**
- **Kenapa k = 5 padahal k = 3 skornya terbaik?** k = 3 hanya membuat kelompok kasar: "aktif" (VIP dan biasa digabung), "tidak aktif" (mulai pergi dan sudah lama pergi digabung), dan "baru". Kelompok kasar tidak bisa diberi aksi berbeda. **k = 5** adalah k terkecil yang memisahkan 4 segmen bisnis **dan** BARU. Kualitasnya setara versi 1 dan cluster terkecilnya masih 12,5 %. Metrik dipakai untuk menyaring, lalu kegunaan bisnis menentukan.
- **Silhouette 0,254: kecil? Kenapa?**

  **Ya, tergolong lemah.** Skala umum (Kaufman & Rousseeuw):

  | Silhouette | Arti |
  |---|---|
  | > 0,70 | struktur kuat |
  | 0,50–0,70 | wajar |
  | 0,25–0,50 | lemah |
  | < 0,25 | hampir tidak ada struktur |

  Nilai kami (0,254) berada di batas bawah "lemah".

  **Penyebabnya:**
  1. **Perilaku belanja adalah spektrum, bukan kelompok terpisah.** Tidak ada garis tegas antara pelanggan yang belanja 3× dan 4×, atau 90 dan 110 hari lalu. Data seperti ini menyebar kontinu, sehingga banyak pelanggan berada di **perbatasan** dua segmen. Silhouette menghukum titik di perbatasan (nilainya mendekati 0).
  2. **Banyak pelanggan menumpuk di area yang sama.** ± 28 % pelanggan hanya belanja sekali dengan nilai kecil, sehingga titiknya padat dan saling berdekatan.
  3. **6 dimensi.** Makin banyak fitur, jarak antar-titik makin "mirip" satu sama lain, dan silhouette cenderung turun.
  4. **Asumsi K-Means.** Cluster dianggap bulat dan berukuran mirip, padahal bentuk data asli memanjang dan miring (masih miring walau sudah log1p).
  5. **Kami sengaja tidak mengejar silhouette maksimum.** k = 3 memang lebih tinggi (0,320), tetapi kurang berguna untuk bisnis (lihat pertanyaan k = 5 di atas).

  **Kenapa tetap layak dipakai:**
  - Segmentasi pelanggan memang dipakai untuk **membagi spektrum menjadi kelompok yang bisa ditindaklanjuti**, bukan menemukan kelompok yang benar-benar terpisah. Silhouette 0,2–0,4 umum di segmentasi RFM data riil.
  - **Stabil:** ARI **0,997** antar 5 seed. Segmen hampir identik walau model diulang, jadi bukan hasil acak.
  - **Profilnya jelas berbeda** (median Recency 17 vs 411 hari, Monetary £ 4.472 vs £ 248).
  - **Validasi eksternal:** peluang churn tiap segmen berurutan 17 % → 84 % tanpa memakai label churn saat clustering.
  - Davies-Bouldin 1,187 dan cluster terkecil 12,5 % (tidak ada cluster "remah").

  **Kalau dosen bertanya cara menaikkannya:** Gaussian Mixture Model (cluster boleh lonjong), mengurangi fitur, atau memakai k = 3. Masing-masing ada konsekuensinya, terutama k = 3 yang menghilangkan segmen bisnis.
- **Kenapa K-Means?** Cepat, mudah dijelaskan (centroid = profil), dan **bisa menempatkan pelanggan baru** ke cluster terdekat, syarat untuk prediksi real-time di web. DBSCAN tidak punya fungsi prediksi untuk data baru; hierarchical berat untuk ribuan data dan juga tidak bisa memprediksi.
- **Kenapa log1p + scaling?** K-Means memakai jarak Euclidean. Tanpa scaling, Monetary (ribuan £) mendominasi; tanpa log, pembeli grosir membentuk cluster sendiri.
- **Nama segmen ditentukan bagaimana?**
  1. Cluster dengan centroid `CustomerAge` termuda diberi nama BARU.
  2. Sisanya diurutkan dengan skor −Recency + Frequency + Monetary + UniqueProducts menjadi LOYAL → POTENSIAL → BERISIKO → HILANG.

  Nama tetap konsisten walaupun nomor cluster berubah saat retrain.

### 5.3 Model 3: Association rules (FP-Growth)
Parameter: min_support **0,5 %**, confidence ≥ 0,3, lift ≥ 2, maksimal 3 produk per rule. Hasilnya **1.731 rules** dengan ± 400 produk pemicu. Contoh: HERB MARKER THYME → HERB MARKER ROSEMARY (confidence 94,8 %); PINK REGENCY TEACUP → ROSES REGENCY TEACUP (lift 34,9).

| min_support | Rules | Produk pemicu | Hit@5 |
|---|---|---|---|
| 1 % | 298 | 127 | 8,4 % |
| 0,75 % | 693 | 227 | 10,0 % |
| **0,5 %** | **1.819** | **402** | **11,8 %** |
| 0,3 % | 5.438 | 737 | 13,1 % |
| *Baseline 5 produk terlaris* | – | – | *2,5 %* |

**Pertanyaan:**
- **Cara evaluasi?** Rules dibuat dari data **sebelum** 90 hari terakhir, lalu diuji pada 5.000 invoice di 90 hari terakhir. Dari tiap invoice, 1 produk disembunyikan dan sisanya diberikan ke sistem; hit bila produk tersembunyi ada di 5 rekomendasi.
- **Kenapa 0,5 %?** Titik seimbang. Itemset masih didukung ± 180 invoice (andal), Hit@5 naik 41 % dari 1 %, dan jumlah rules ringan untuk web. Turun ke 0,3 % hanya menambah 1,3 poin, tetapi rules naik 3×.
- **11,8 % kecil?** Dibanding acak (≈ 0,2 %) dan produk terlaris (2,5 %), hasil ini 4,7× lebih baik. Rule aktif di 85 % invoice, tetapi hanya bisa merekomendasikan ± 340 produk. Bila produk yang dicari termasuk produk itu, tebakan benar **42 %**.
- **Kenapa FP-Growth?** Hasilnya identik dengan Apriori, tetapi jauh lebih cepat untuk matriks 36.280 × 2.765.
- **Untuk apa?** Cross-sell di kasir, isi voucher, dan bundling. Fallback ke produk terlaris bila tidak ada rule yang cocok.

### 5.4 Kenapa metriknya itu?
Setiap jenis model punya metrik berbeda karena **pertanyaannya berbeda**.

**Classification (churn): ada label, jadi bisa dihitung benar/salah**

| Metrik | Peran | Kenapa dipakai / tidak |
|---|---|---|
| **ROC-AUC** | **Utama** (pemilihan model, laporan) | Web memakai **probabilitas** untuk mengurutkan pelanggan paling berisiko. ROC-AUC menilai kualitas **urutan** itu dan **tidak bergantung threshold**. Juga adil saat kelas sedikit tidak seimbang. |
| Precision | Pendukung | "Dari yang kita hubungi, berapa yang benar berisiko?" Biaya alarm palsu = voucher terbuang. |
| Recall | Pendukung | "Dari yang benar akan pergi, berapa yang tertangkap?" Biaya terlewat = kehilangan pelanggan. |
| F1 | Pendukung | Menyeimbangkan precision dan recall dalam satu angka. |
| Accuracy | Hanya dilaporkan | **Menyesatkan**: Dummy yang selalu menebak "churn" sudah 56,6 % padahal tidak berguna. |
| Confusion matrix | Diagnosis | Melihat jenis kesalahan (162 alarm palsu vs 179 terlewat). |
| Churn aktual per level risiko | Validasi bisnis | Membuktikan RENDAH/SEDANG/TINGGI benar-benar berurutan (21 % → 55 % → 84 %). |
| *PR-AUC, log-loss* | Tidak dipakai utama | PR-AUC berguna saat kelas positif sangat langka (mis. 5 %); churn kami 56,6 %. Log-loss menilai kalibrasi, sedangkan fokus kami urutan prioritas. |

**Clustering (segmen): tidak ada label, jadi dinilai dari bentuk cluster**

| Metrik | Mengukur | Kenapa dipakai |
|---|---|---|
| Inertia | Kerapatan dalam cluster | Grafik elbow; tidak bisa memilih k sendirian karena selalu turun |
| **Silhouette** | Rapat ke cluster sendiri **vs** jauh dari cluster lain | Metrik internal paling umum, skala −1..1 mudah dibaca |
| **Davies-Bouldin** | Sebaran ÷ jarak antar-pusat | Pembanding silhouette dari sudut lain; dua metrik yang sepakat lebih meyakinkan |
| Cluster terkecil (%) | Ukuran segmen | Segmen < 5 % terlalu kecil untuk kampanye pemasaran |
| **ARI antar-seed** | Stabilitas | K-Means bergantung titik awal acak; ARI ≈ 1 membuktikan hasil tidak kebetulan |
| *Calinski-Harabasz* | Rasio sebaran antar/dalam cluster | Tidak dipakai: cenderung memihak k besar dan sejalan dengan DBI |

**Association rules (rekomendasi): dua tahap**

| Tahap | Metrik | Kenapa |
|---|---|---|
| **Menyaring rule** | support ≥ 0,5 % | Pola harus muncul cukup sering (± 180 invoice), bukan kebetulan |
| | confidence ≥ 0,3 | Minimal 30 % pembeli A juga membeli B, agar rekomendasi cukup sering tepat |
| | lift ≥ 2 | B dibeli bersama A minimal 2× lebih sering daripada biasanya. Mencegah rule "palsu" yang muncul hanya karena B memang produk laris |
| | max 3 produk per rule | Rule panjang jarang terpenuhi dan sulit dijelaskan |
| **Menilai kegunaan** | **Hit Rate@5** vs baseline | Support/confidence/lift hanya menilai rule di data latih. Hit@5 menguji langsung "apakah rekomendasinya tepat untuk transaksi **masa depan**". Angka **5** karena web menampilkan 5 rekomendasi. |

---

## 6. Kenapa Masih Ada yang Salah?

**Churn (salah 25,9 %).** Kesalahan menumpuk di pelanggan "zona abu-abu":

| Kelompok | Error |
|---|---|
| Recency > 400 hari | 10 % |
| Frequency > 10× | 11,6 % |
| Recency 31–180 hari | 33–37 % |
| Model ragu (probabilitas 0,4–0,7) | **39,9 %** |
| Model yakin | **18,1 %** |

Penyebab:
- Data tidak memuat alasan pelanggan pergi (kepuasan, pesaing, promo).
- Ada pembeli musiman dan grosir yang ritmenya tidak teratur.
- Batas 90 hari bersifat tegas: kembali di hari ke-91 tetap dihitung churn.
- Riwayat pelanggan yang hanya 1–3 transaksi terlalu sedikit.
- Periode label mencakup musim Natal.

**Segmentasi.** Batas antar-segmen tidak tajam; pelanggan di perbatasan bisa masuk segmen tetangga.

**Rekomendasi.** Banyak produk jarang dibeli bersama produk lain, sehingga tidak punya rule. Keranjang juga sangat beragam (median 17 produk).

---

## 7. Menyimpan Model & Deployment

```
models/
├── current.json                 ← versi aktif
└── 20261111-003623/
    ├── churn_model.joblib       Pipeline log1p + RandomForest (± 2,6 MB)
    ├── segment_model.joblib     Pipeline log1p + scaler + KMeans (± 25 KB)
    ├── association_rules.json   1.731 rules (± 356 KB)
    └── metadata.json            fitur, threshold, nama segmen, metrik, batas capping, versi library
```

**Kenapa joblib?**
1. **Standar resmi scikit-learn** untuk menyimpan model; satu baris `joblib.dump` / `joblib.load`.
2. **Efisien untuk array numpy besar.** Random Forest berisi 300 pohon dengan ribuan array; joblib menyimpannya lebih cepat dan ringkas daripada `pickle` biasa (mendukung kompresi).
3. **Menyimpan seluruh Pipeline**, bukan hanya model. `log1p`, scaler, dan model ikut tersimpan, sehingga preprocessing di web **dijamin identik** dengan saat training. Tidak ada rumus yang ditulis ulang.
4. **Catatan keamanan dan versi:**
   - joblib berbasis pickle, jadi hanya file dari sumber tepercaya yang boleh dimuat.
   - Versi scikit-learn saat menyimpan dan memuat harus sama. Karena itu versi library dikunci dan dicatat di `metadata.json`, dan notebook mengecek versi di sel pertama.

**Kenapa rules disimpan sebagai JSON, bukan joblib?** Rules hanya berupa daftar teks dan angka. JSON bisa dibaca manusia, mudah disalin ke tabel database `ml_association_rules`, dan bisa dibaca PHP/Laravel langsung untuk fallback.

**Kenapa ada `metadata.json`?** Model sendiri tidak tahu urutan fiturnya, threshold, atau nama segmen. Metadata menyimpan semua itu agar ML service dan web memakai konfigurasi yang sama, dan agar versi model bisa dilacak dan di-rollback.

**Alur di web:**
- **Invoice baru:** profil pelanggan langsung dihitung ulang. Contoh #12346: SEDANG 67,8 % BERISIKO → 61,3 % POTENSIAL.
- **Scoring tiap malam:** dilakukan karena Recency bertambah setiap hari.
- **Retrain:** rules mingguan, churn dan segmen bulanan. Ada *quality gate*: model baru aktif hanya bila ROC-AUC ≥ 0,70 dan tidak lebih buruk > 0,02 dari model lama.
- **Kenapa tidak retrain tiap malam?** Label churn butuh 90 hari data masa depan, sehingga data berlabel baru per hari hampir nol.

---

## 8. Pertanyaan Kritis Lain

- **Keterbatasan?**
  - Data hanya 2 tahun dan 85 % dari UK.
  - Tidak ada demografi pelanggan.
  - Silhouette sedang.
  - Rules hanya mencakup ± 400 produk.
  - Kenaikan churn dari fitur baru kecil (+0,005).
- **Kenapa kenaikan 0,005 dianggap berarti?** Konsisten di 20 dari 20 fold pada pembagian data identik. Kalau hanya kebetulan, hasilnya akan naik-turun.
- **Inertia vs Davies-Bouldin?** Inertia hanya mengukur kerapatan dan selalu turun bila k naik, jadi hanya dipakai untuk elbow. DBI mengukur kerapatan **dan** keterpisahan, sehingga bisa dipakai membandingkan k.
- **Z-score di heatmap segmen?** Posisi rata-rata segmen relatif terhadap semua pelanggan dalam satuan std. Ciri khas tiap segmen terlihat walau satuan fitur berbeda. Contoh: BARU dan HILANG sama-sama rendah di Frequency, tetapi BARU sangat rendah di CustomerAge.
- **Pengembangan berikutnya?**
  - Churn: fitur tren 30/60/90 hari dan musim, survival analysis, kalibrasi probabilitas.
  - Segmentasi: Gaussian Mixture.
  - Rekomendasi: collaborative filtering untuk produk tanpa rule.

### Kalimat penutup untuk dosen
> Label churn dibuat secara temporal agar tidak ada leakage. Setiap model dibandingkan dengan baseline dan dipilih lewat cross-validation. Perbaikan versi 2 (CustomerAge) dibuktikan dengan perbandingan berpasangan, bukan klaim: ROC-AUC naik di 20/20 fold, dan pelanggan baru tidak lagi salah masuk segmen HILANG.
