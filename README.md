# BACE-1 Inhibitor Predictor

Aplikasi web prediksi senyawa inhibitor **BACE-1** (β-site Amyloid Precursor Protein Cleaving Enzyme 1) berbasis  **Graph Neural Network (GNN)** , dibangun dengan Streamlit.

---

## Deskripsi

BACE-1 adalah target terapi utama dalam pengembangan obat Alzheimer. Aplikasi ini mengklasifikasikan apakah suatu senyawa kimia berpotensi sebagai inhibitor BACE-1 berdasarkan struktur molekulnya dalam format  **SMILES** .

---

## Fitur

* **Single SMILES** — prediksi satu senyawa dengan visualisasi struktur 2D dan scaffold Murcko
* **Batch SMILES** — prediksi banyak senyawa sekaligus via CSV atau text area (maks 100)
* **Compare Models** — bandingkan prediksi 4 varian model untuk satu senyawa
* **About** — dokumentasi model, dataset, dan interpretasi hasil
* **Dark / Light theme** toggle
* **Out-of-domain warning** otomatis jika senyawa di luar rentang training

---

## Model

4 varian model tersedia, masing-masing merupakan  **ensemble 5 seed** :

| Variant             | Arsitektur                  | Split Strategy        |
| ------------------- | --------------------------- | --------------------- |
| `gin_scaffold`    | GIN (tanpa edge features)   | Scaffold              |
| `gin_stratified`  | GIN                         | Stratified            |
| `gine_scaffold`   | GINE (dengan edge features) | Scaffold              |
| `gine_stratified` | GINE                        | Stratified*(default)* |

**Default: `gine_stratified`** — PR-AUC tertinggi.

---

## Dataset

* Sumber:  **ChEMBL v36** , target BACE-1 (CHEMBL4822)
* Jumlah: **7.829 senyawa** setelah preprocessing
* Threshold: IC50 ≤ 1 μM (pIC50 ≥ 6.0) → kelas aktif
* Split: Train 70% / Val 15% / Test 15%

---

## Struktur Folder

```
webapp_v2/
├── app.py              # Streamlit UI utama
├── inference.py        # MultiModelPredictor (ensemble inference)
├── model_def.py        # Arsitektur GIN & GINE
├── molecular_utils.py  # Featurisasi & preprocessing SMILES
├── requirements.txt    # Dependensi Python
└── models/
    ├── gin/
    │   ├── scaffold/
    │   │   ├── gin_best_model.pt
    │   │   └── gin_seed_*.pt   (5 files)
    │   └── stratified/
    │       ├── gin_best_model.pt
    │       └── gin_seed_*.pt
    └── gine/
        ├── scaffold/
        │   ├── gine_best_model.pt
        │   └── gine_seed_*.pt
        └── stratified/
            ├── gine_best_model.pt
            └── gine_seed_*.pt
```

---

## Instalasi & Menjalankan Lokal

```bash
# 1. Clone repo
git clone https://github.com/username/nama-repo.git
cd nama-repo

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan aplikasi
streamlit run app.py
```

---


## Interpretasi Hasil

| Output                  | Keterangan                                                           |
| ----------------------- | -------------------------------------------------------------------- |
| **P(active)**     | Probabilitas senyawa aktif (0–1), threshold 0.5                     |
| **σ ensemble**   | Std deviasi antar 5 model; nilai > 0.10 = disagreement tinggi        |
| **Confidence**    | High / Medium / Low berdasarkan P dan σ                             |
| **Out-of-domain** | Muncul jika MW, jumlah atom, atau atom type di luar rentang training |

---

## Keterbatasan

* Hanya memprediksi aktivitas terhadap BACE-1, bukan toksisitas atau ADMET
* Prediksi bersifat  **computational** , tidak menggantikan uji laboratorium
* Performa scaffold split lebih realistis untuk drug discovery dibanding stratified split

---

## Referensi

* Xu et al. (2019) —  *How Powerful are Graph Neural Networks?* , ICLR
* Hu et al. (2020) —  *Strategies for Pre-training Graph Neural Networks* , ICLR
* ChEMBL Database — BACE-1 inhibitors (CHEMBL4822)
* Frameworks: PyTorch, PyTorch Geometric, RDKit, Streamlit
