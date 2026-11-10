"""Detektor bahasa daerah Indonesia.

Contoh penggunaan:

    from bahasa_detector import DetektorBahasa

    detektor = DetektorBahasa.dari_file()
    hasil = detektor.prediksi("Aku arep menyang pasar tuku sayur.")
    print(hasil["bahasa"], hasil["keyakinan"])
"""

from .config import MODEL_PATH, WILAYAH
from .dataset import muat_data
from .model import DetektorBahasa, buat_pipeline
from .teks import normalisasi
from .terjemah import Penerjemah

__version__ = "1.1.0"

__all__ = [
    "DetektorBahasa",
    "MODEL_PATH",
    "Penerjemah",
    "WILAYAH",
    "buat_pipeline",
    "muat_data",
    "normalisasi",
]
