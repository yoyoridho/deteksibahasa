"""Pembersihan dan normalisasi teks."""

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"[@#]\w+")
_BUKAN_HURUF = re.compile(r"[^a-zà-öø-ÿ\s']")
_SPASI = re.compile(r"\s+")


def normalisasi(teks: str) -> str:
    """Ubah teks menjadi bentuk yang konsisten untuk model.

    - Unicode diseragamkan (NFKC) dan huruf diubah ke kecil.
    - URL, mention, hashtag, angka, dan tanda baca dihapus.
    - Apostrof dipertahankan karena dipakai di beberapa bahasa daerah.
    - Spasi berlebih dirapikan.
    """
    if not isinstance(teks, str):
        raise TypeError("Teks harus berupa string.")

    teks = unicodedata.normalize("NFKC", teks).lower()
    teks = _URL.sub(" ", teks)
    teks = _MENTION.sub(" ", teks)
    teks = _BUKAN_HURUF.sub(" ", teks)
    teks = _SPASI.sub(" ", teks)
    return teks.strip()


def normalisasi_banyak(daftar_teks) -> list[str]:
    """Normalisasi sekumpulan teks (dipakai di dalam pipeline scikit-learn)."""
    return [normalisasi(teks) for teks in daftar_teks]
