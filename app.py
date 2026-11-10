"""Web app deteksi bahasa daerah.

Jalankan:
    python app.py
Lalu buka http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import os

from flask import Flask, jsonify, render_template, request

from bahasa_detector.config import MODEL_PATH, TERJEMAHAN_ONLINE
from bahasa_detector.model import DetektorBahasa
from bahasa_detector.terjemah import Penerjemah

app = Flask(__name__)
app.json.ensure_ascii = False

_detektor: DetektorBahasa | None = None
_penerjemah: Penerjemah | None = None


def dapatkan_detektor() -> DetektorBahasa:
    """Muat model sekali saja (lazy loading)."""
    global _detektor
    if _detektor is None:
        _detektor = DetektorBahasa.dari_file(MODEL_PATH)
    return _detektor


def dapatkan_penerjemah() -> Penerjemah:
    """Muat mesin terjemahan sekali saja (lazy loading)."""
    global _penerjemah
    if _penerjemah is None:
        _penerjemah = Penerjemah()
    return _penerjemah


def info_model() -> dict:
    """Informasi model untuk halaman dan endpoint /api/info."""
    if not MODEL_PATH.exists():
        return {
            "siap": False,
            "pesan": "Model belum dilatih. Jalankan: python train.py",
            "n_data": 0,
            "jumlah_kelas": 0,
            "akurasi_cv": None,
            "akurasi_holdout": None,
            "dibuat": None,
            "nama_model": "-",
        }

    metadata = dapatkan_detektor().metadata
    return {
        "siap": True,
        "pesan": "",
        "n_data": metadata.get("n_data", 0),
        "jumlah_kelas": metadata.get("jumlah_kelas", 0),
        "akurasi_cv": metadata.get("akurasi_cv"),
        "akurasi_holdout": metadata.get("akurasi_holdout"),
        "dibuat": metadata.get("dibuat"),
        "nama_model": metadata.get("nama_model", "-"),
    }


@app.get("/")
def beranda():
    return render_template(
        "index.html", info=info_model(), terjemahan_online=TERJEMAHAN_ONLINE
    )


@app.get("/api/info")
def api_info():
    return jsonify(info_model())


@app.get("/api/sehat")
def api_sehat():
    return jsonify({"status": "ok"})


@app.post("/api/deteksi")
def api_deteksi():
    data = request.get_json(silent=True) or {}
    teks = (data.get("teks") or "").strip()

    if not teks:
        return jsonify({"error": "Teks tidak boleh kosong."}), 400
    if len(teks) > 5000:
        return jsonify({"error": "Teks terlalu panjang (maksimal 5000 karakter)."}), 400

    try:
        hasil = dapatkan_detektor().prediksi(teks)
    except FileNotFoundError as kesalahan:
        return jsonify({"error": str(kesalahan)}), 503
    except ValueError as kesalahan:
        return jsonify({"error": str(kesalahan)}), 400
    except Exception:  # pragma: no cover - jaring pengaman untuk server
        app.logger.exception("Gagal melakukan deteksi")
        return jsonify({"error": "Terjadi kesalahan internal pada server."}), 500

    return jsonify(hasil)


@app.post("/api/terjemahkan")
def api_terjemahkan():
    data = request.get_json(silent=True) or {}
    teks = (data.get("teks") or "").strip()

    if not teks:
        return jsonify({"error": "Teks tidak boleh kosong."}), 400
    if len(teks) > 5000:
        return jsonify({"error": "Teks terlalu panjang (maksimal 5000 karakter)."}), 400

    bahasa = (data.get("bahasa") or "").strip() or None
    online = data.get("online")
    if isinstance(online, str):
        online = online.strip().lower() not in {"0", "false", "off", "tidak", "no"}
    elif online is not None:
        online = bool(online)

    try:
        detektor = dapatkan_detektor() if bahasa is None else None
        hasil = dapatkan_penerjemah().terjemahkan(
            teks, bahasa=bahasa, online=online, detektor=detektor
        )
    except FileNotFoundError as kesalahan:
        return jsonify({"error": str(kesalahan)}), 503
    except ValueError as kesalahan:
        return jsonify({"error": str(kesalahan)}), 400
    except Exception:  # pragma: no cover - jaring pengaman untuk server
        app.logger.exception("Gagal menerjemahkan teks")
        return jsonify({"error": "Terjadi kesalahan internal pada server."}), 500

    return jsonify(hasil)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jalankan web app deteksi bahasa daerah.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Detektor Bahasa Daerah berjalan di http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
