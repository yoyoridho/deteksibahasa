"""Konfigurasi global untuk proyek deteksi bahasa daerah."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "korpus.csv"
KAMUS_DIR = BASE_DIR / "data" / "kamus"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model_bahasa.joblib"
REPORT_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_CONFIDENCE = 0.30

# --- Terjemahan ke bahasa Indonesia ---------------------------------------
# Google Translate dipakai lebih dulu untuk bahasa yang didukung. Jika
# dinonaktifkan/gagal, terjemahan memakai korpus paralel lalu kamus lokal.
TERJEMAHAN_ONLINE = True
TERJEMAHAN_TIMEOUT = 8
AMBANG_KORPUS = 0.72
MIN_IRISAN_KORPUS = 0.55
MIN_CAKUPAN_KAMUS = 0.45

# Kode bahasa Google Translate. Bugis dan Banjar belum didukung Google,
# sehingga selalu memakai mesin lokal.
KODE_GOOGLE = {
    "Jawa": "jw",
    "Sunda": "su",
    "Minang": "min",
    "Batak": "bbc",
    "Bugis": None,
    "Bali": "ban",
    "Madura": "mad",
    "Banjar": None,
    "Indonesia": None,
}

# Informasi wilayah untuk ditampilkan di CLI dan web.
WILAYAH = {
    "Jawa": "Jawa Tengah, Yogyakarta, Jawa Timur",
    "Sunda": "Jawa Barat, Banten",
    "Minang": "Sumatera Barat",
    "Batak": "Sumatera Utara (Batak Toba)",
    "Bugis": "Sulawesi Selatan",
    "Bali": "Bali",
    "Madura": "Jawa Timur (Pulau Madura)",
    "Banjar": "Kalimantan Selatan",
    "Indonesia": "Bahasa Indonesia (bahasa nasional)",
}

DESKRIPSI_MODEL = "TF-IDF karakter + kata, Logistic Regression"
