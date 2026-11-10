"""Pembacaan dan validasi korpus."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .config import DATA_PATH


def muat_data(path: str | Path = DATA_PATH) -> tuple[list[str], list[str], list[str]]:
    """Muat korpus CSV dan kembalikan (teks, label, grup).

    Kolom wajib: ``bahasa`` dan ``teks``. Kolom ``grup`` bersifat opsional;
    grup dipakai agar kalimat paralel tidak muncul di data latih dan data uji
    secara bersamaan.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Berkas korpus tidak ditemukan: {path}")

    teks: list[str] = []
    label: list[str] = []
    grup: list[str] = []

    with path.open("r", encoding="utf-8-sig", newline="") as berkas:
        pembaca = csv.DictReader(berkas)
        kolom = set(pembaca.fieldnames or [])
        if not {"bahasa", "teks"}.issubset(kolom):
            raise ValueError("Korpus harus memiliki kolom 'bahasa' dan 'teks'.")

        for nomor, baris in enumerate(pembaca, start=2):
            isi = (baris.get("teks") or "").strip()
            bahasa = (baris.get("bahasa") or "").strip()
            if not isi or not bahasa:
                continue
            teks.append(isi)
            label.append(bahasa)
            grup.append((baris.get("grup") or "").strip() or f"baris-{nomor}")

    if len(teks) < 10:
        raise ValueError("Korpus terlalu kecil; minimal 10 baris teks.")
    if len(set(label)) < 2:
        raise ValueError("Korpus harus memiliki minimal dua kelas bahasa.")

    hitung = Counter(label)
    terlalu_sedikit = [b for b, n in hitung.items() if n < 5]
    if terlalu_sedikit:
        raise ValueError(
            "Setiap bahasa minimal butuh 5 contoh. Kurang: "
            + ", ".join(sorted(terlalu_sedikit))
        )

    return teks, label, grup


def ringkasan_kelas(label: list[str]) -> dict[str, int]:
    """Kembalikan jumlah contoh per bahasa, diurutkan berdasarkan nama."""
    return dict(sorted(Counter(label).items()))
