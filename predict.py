"""Deteksi bahasa daerah dari teks melalui command line.

Contoh:
    python predict.py "Aku arep menyang pasar tuku sayur."
    python predict.py --file contoh.txt
    python predict.py --interaktif
    python predict.py --json "Ulun handak tulak ka pasar."
    python predict.py --terjemahkan "Iyya lao ri pasa macari kaju."
    python predict.py --terjemahkan --offline "Ulun handak tulak ka pasar."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bahasa_detector.config import MODEL_PATH
from bahasa_detector.model import DetektorBahasa


def tampilkan(hasil: dict, top: int, terjemahan: dict | None = None) -> None:
    print("\n" + "-" * 60)
    print(f"Teks       : {hasil['teks']}")
    print(f"Bahasa     : {hasil['bahasa']}")
    print(f"Wilayah    : {hasil['wilayah']}")
    print(f"Keyakinan  : {hasil['keyakinan']:.1%}")
    if hasil["kata_kunci"]:
        print(f"Kata kunci : {', '.join(hasil['kata_kunci'])}")
    if hasil["peringatan"]:
        print(f"Catatan    : {hasil['peringatan']}")
    if terjemahan:
        print(f"Terjemahan : {terjemahan['terjemahan']}")
        print(f"Metode     : {terjemahan['metode']}")
        if terjemahan.get("catatan"):
            print(f"Info       : {terjemahan['catatan']}")
    print(f"\nTop {top}:")
    for nomor, item in enumerate(hasil["top"], start=1):
        print(
            f"  {nomor}. {item['bahasa']:<10} {item['keyakinan']:>6.1%}  "
            f"({item['wilayah']})"
        )
    print("-" * 60)


def mode_interaktif(
    detektor: DetektorBahasa, top: int, penerjemah=None, online: bool = True
) -> None:
    print("Mode interaktif. Ketik teks lalu tekan Enter.")
    print("Ketik 'keluar' untuk berhenti.\n")
    while True:
        try:
            teks = input("Teks> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSelesai.")
            break
        if teks.lower() in {"keluar", "exit", "quit"}:
            print("Selesai.")
            break
        if not teks:
            continue
        try:
            hasil = detektor.prediksi(teks, top_k=top)
            terjemahan = None
            if penerjemah is not None:
                terjemahan = penerjemah.terjemahkan(
                    teks, bahasa=hasil["bahasa"], online=online
                )
            tampilkan(hasil, top, terjemahan)
        except ValueError as kesalahan:
            print(f"Kesalahan: {kesalahan}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deteksi bahasa daerah dari teks.")
    parser.add_argument("teks", nargs="*", help="Teks yang akan dideteksi.")
    parser.add_argument("--file", type=Path, help="Baca teks dari berkas.")
    parser.add_argument("--interaktif", "-i", action="store_true", help="Mode interaktif.")
    parser.add_argument("--json", action="store_true", help="Keluarkan hasil sebagai JSON.")
    parser.add_argument("--top", type=int, default=3, help="Jumlah kandidat bahasa (default 3).")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Berkas model joblib.")
    parser.add_argument(
        "--terjemahkan",
        action="store_true",
        help="Tampilkan terjemahan bahasa Indonesia.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Paksa terjemahan offline (tanpa Google Translate).",
    )
    args = parser.parse_args()

    try:
        detektor = DetektorBahasa.dari_file(args.model)
    except FileNotFoundError as kesalahan:
        print(f"Kesalahan: {kesalahan}", file=sys.stderr)
        raise SystemExit(1)

    penerjemah = None
    if args.terjemahkan:
        from bahasa_detector.terjemah import Penerjemah

        penerjemah = Penerjemah(online=not args.offline)

    if args.interaktif:
        mode_interaktif(detektor, args.top, penerjemah, online=not args.offline)
        return

    if args.file:
        if not args.file.exists():
            print(f"Kesalahan: berkas tidak ditemukan: {args.file}", file=sys.stderr)
            raise SystemExit(1)
        teks = args.file.read_text(encoding="utf-8-sig").strip()
    elif args.teks:
        teks = " ".join(args.teks)
    else:
        parser.print_help()
        print('\nContoh: python predict.py "Aku arep menyang pasar tuku sayur."')
        return

    try:
        hasil = detektor.prediksi(teks, top_k=args.top)
        terjemahan = None
        if penerjemah is not None:
            terjemahan = penerjemah.terjemahkan(
                teks, bahasa=hasil["bahasa"], online=not args.offline
            )
    except ValueError as kesalahan:
        print(f"Kesalahan: {kesalahan}", file=sys.stderr)
        raise SystemExit(1)

    if args.json:
        keluaran = dict(hasil)
        if terjemahan is not None:
            keluaran["terjemahan"] = terjemahan
        print(json.dumps(keluaran, ensure_ascii=False, indent=2))
    else:
        tampilkan(hasil, args.top, terjemahan)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
