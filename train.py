"""Latih model deteksi bahasa daerah.

Contoh:
    python train.py
    python train.py --data data/korpus.csv --model models/model_bahasa.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bahasa_detector.config import DATA_PATH, MODEL_PATH, REPORT_DIR, TEST_SIZE
from bahasa_detector.dataset import muat_data, ringkasan_kelas
from bahasa_detector.latih import latih_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih model deteksi bahasa daerah.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Berkas korpus CSV.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Lokasi model keluaran.")
    parser.add_argument("--reports", type=Path, default=REPORT_DIR, help="Folder laporan evaluasi.")
    parser.add_argument(
        "--test-size", type=float, default=TEST_SIZE, help="Proporsi data uji (default 0.2)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--C",
        dest="daftar_C",
        nargs="+",
        type=float,
        default=[1.0, 3.0, 8.0, 15.0],
        help="Kandidat nilai regularisasi C.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PELATIHAN DETEKTOR BAHASA DAERAH")
    print("=" * 60)

    teks, label, grup = muat_data(args.data)
    distribusi = ringkasan_kelas(label)
    print(f"Korpus      : {args.data}")
    print(f"Jumlah data : {len(teks)} kalimat")
    print(f"Kelas       : {len(distribusi)}")
    print(f"Grup        : {len(set(grup))}")
    print("\nDistribusi kelas:")
    for bahasa, jumlah in distribusi.items():
        print(f"  {bahasa:<12} {jumlah:>4}")

    print("\nMelatih model (TF-IDF + Logistic Regression)...")
    hasil = latih_model(
        data_path=args.data,
        model_path=args.model,
        report_dir=args.reports,
        test_size=args.test_size,
        random_state=args.seed,
        daftar_C=tuple(args.daftar_C),
    )

    metadata = hasil["metadata"]
    print("\n" + "=" * 60)
    print("HASIL")
    print("=" * 60)
    print(f"Parameter C terbaik : {metadata['parameter']['C']}")
    print(f"Akurasi hold-out    : {hasil['akurasi_holdout']:.2%}")
    print(f"Macro F1 hold-out   : {hasil['macro_f1_holdout']:.4f}")
    print(f"Akurasi CV 5 lipat  : {hasil['akurasi_cv']:.2%}")
    print(f"Macro F1 CV         : {hasil['macro_f1_cv']:.4f}")
    print(f"\nModel disimpan      : {hasil['model_path']}")
    print(f"Laporan evaluasi    : {Path(hasil['report_dir']) / 'laporan_evaluasi.txt'}")
    print(f"Metrik JSON         : {Path(hasil['report_dir']) / 'metrics.json'}")

    print("\nLaporan klasifikasi hold-out:")
    print(hasil["laporan_holdout"])

    print("Contoh pemakaian:")
    print('  python predict.py "Aku arep menyang pasar tuku sayur."')
    print("  python app.py  # lalu buka http://127.0.0.1:5000")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
