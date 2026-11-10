"""Pelatihan, validasi, dan penyimpanan model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict

from .config import (
    DATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
)
from .dataset import muat_data, ringkasan_kelas
from .model import DetektorBahasa, buat_pipeline, metadata_dasar


def _bagi_grup(
    X: list[str],
    y: list[str],
    grup: list[str],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bagi data per grup agar kalimat paralel tidak bocor ke data uji."""
    n_splits = max(2, round(1 / test_size))
    pembagi = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    latih, uji = next(pembagi.split(X, y, grup))
    return latih, uji


def _format_matriks(matriks: np.ndarray, kelas: list[str]) -> str:
    lebar = max(len(k) for k in kelas) + 2
    lebar_kolom = max(7, max(len(k) for k in kelas) + 2)
    baris = [" " * lebar + "".join(k.rjust(lebar_kolom) for k in kelas)]
    for nama, nilai in zip(kelas, matriks):
        baris.append(
            nama.ljust(lebar)
            + "".join(str(int(v)).rjust(lebar_kolom) for v in nilai)
        )
    return "\n".join(baris)


def latih_model(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    report_dir: str | Path = REPORT_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    daftar_C: tuple[float, ...] = (1.0, 3.0, 8.0, 15.0),
) -> dict[str, Any]:
    """Latih model, evaluasi, simpan model, dan tulis laporan.

    Mengembalikan ringkasan hasil pelatihan.
    """
    X, y, grup = muat_data(data_path)
    kelas = sorted(set(y))
    distribusi = ringkasan_kelas(y)

    idx_latih, idx_uji = _bagi_grup(X, y, grup, test_size, random_state)
    X_latih = [X[i] for i in idx_latih]
    y_latih = [y[i] for i in idx_latih]
    g_latih = [grup[i] for i in idx_latih]
    X_uji = [X[i] for i in idx_uji]
    y_uji = [y[i] for i in idx_uji]

    # Validasi silang internal untuk memilih C. Grup tetap dijaga terpisah.
    cv_internal = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=random_state
    )
    pencari = GridSearchCV(
        buat_pipeline(),
        {"klasifier__C": list(daftar_C)},
        cv=cv_internal,
        scoring="f1_macro",
        n_jobs=1,
        refit=True,
    )
    pencari.fit(X_latih, y_latih, groups=g_latih)
    model_terbaik = pencari.best_estimator_
    C_terbaik = float(pencari.best_params_["klasifier__C"])

    # Evaluasi hold-out (data yang tidak dipakai memilih parameter).
    y_pred_uji = model_terbaik.predict(X_uji)
    akurasi_holdout = accuracy_score(y_uji, y_pred_uji)
    f1_holdout = f1_score(y_uji, y_pred_uji, average="macro", zero_division=0)
    laporan_holdout = classification_report(
        y_uji,
        y_pred_uji,
        labels=kelas,
        target_names=kelas,
        digits=3,
        zero_division=0,
    )
    matriks = confusion_matrix(y_uji, y_pred_uji, labels=kelas)

    # Validasi silang 5 lipat pada seluruh data untuk estimasi yang lebih stabil.
    cv_luar = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    y_pred_cv = cross_val_predict(
        model_terbaik, X, y, groups=grup, cv=cv_luar, n_jobs=1
    )
    akurasi_cv = accuracy_score(y, y_pred_cv)
    f1_cv = f1_score(y, y_pred_cv, average="macro", zero_division=0)
    laporan_cv = classification_report(
        y,
        y_pred_cv,
        labels=kelas,
        target_names=kelas,
        digits=3,
        zero_division=0,
    )

    # Model final dilatih ulang memakai seluruh data.
    model_terbaik.fit(X, y)

    metadata = metadata_dasar(
        n_data=len(X),
        distribusi=distribusi,
        C=C_terbaik,
        akurasi_holdout=akurasi_holdout,
        macro_f1_holdout=f1_holdout,
        akurasi_cv=akurasi_cv,
        macro_f1_cv=f1_cv,
    )
    metadata["n_latih"] = len(X_latih)
    metadata["n_uji"] = len(X_uji)
    metadata["n_grup"] = len(set(grup))

    detektor = DetektorBahasa(pipeline=model_terbaik, metadata=metadata)
    detektor.simpan(model_path)

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    teks_laporan = "\n".join(
        [
            "LAPORAN EVALUASI DETEKSI BAHASA DAERAH",
            "=" * 52,
            f"Model            : {metadata['nama_model']}",
            f"Dibuat           : {metadata['dibuat']}",
            f"Data             : {len(X)} kalimat, {len(kelas)} kelas, {len(set(grup))} grup",
            f"Data latih/uji   : {len(X_latih)} / {len(X_uji)} kalimat",
            f"Parameter C      : {C_terbaik}",
            "",
            "Distribusi kelas:",
            *[f"  {nama:<12} {jumlah:>4}" for nama, jumlah in distribusi.items()],
            "",
            "HASIL HOLD-OUT (20%, dipisah per grup)",
            "-" * 52,
            f"Akurasi          : {akurasi_holdout:.4f}",
            f"Macro F1         : {f1_holdout:.4f}",
            "",
            laporan_holdout,
            "Matriks kebingungan (baris = aktual, kolom = prediksi):",
            _format_matriks(matriks, kelas),
            "",
            "VALIDASI SILANG 5 LIPAT (seluruh data)",
            "-" * 52,
            f"Akurasi          : {akurasi_cv:.4f}",
            f"Macro F1         : {f1_cv:.4f}",
            "",
            laporan_cv,
            "",
            "Catatan: korpus ini adalah data awal yang disusun untuk demonstrasi.",
            "Ganti dengan korpus yang lebih besar dan beragam untuk penggunaan nyata.",
        ]
    )
    (report_dir / "laporan_evaluasi.txt").write_text(teks_laporan, encoding="utf-8")

    metrik = {
        "model": metadata["nama_model"],
        "dibuat": metadata["dibuat"],
        "n_data": len(X),
        "n_latih": len(X_latih),
        "n_uji": len(X_uji),
        "jumlah_kelas": len(kelas),
        "distribusi": distribusi,
        "parameter": metadata["parameter"],
        "holdout": {
            "akurasi": round(float(akurasi_holdout), 6),
            "macro_f1": round(float(f1_holdout), 6),
            "laporan": laporan_holdout,
        },
        "cross_validation": {
            "lipatan": 5,
            "akurasi": round(float(akurasi_cv), 6),
            "macro_f1": round(float(f1_cv), 6),
            "laporan": laporan_cv,
        },
        "matriks_kebingungan": {
            "label": kelas,
            "nilai": matriks.astype(int).tolist(),
        },
        "versi_sklearn": metadata["versi_sklearn"],
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrik, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "metadata": metadata,
        "akurasi_holdout": float(akurasi_holdout),
        "macro_f1_holdout": float(f1_holdout),
        "akurasi_cv": float(akurasi_cv),
        "macro_f1_cv": float(f1_cv),
        "laporan_holdout": laporan_holdout,
        "laporan_cv": laporan_cv,
        "matriks": matriks.tolist(),
        "kelas": kelas,
        "model_path": str(model_path),
        "report_dir": str(report_dir),
    }
