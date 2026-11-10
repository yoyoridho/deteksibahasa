"""Model deteksi bahasa daerah."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .config import DESKRIPSI_MODEL, MIN_CONFIDENCE, MODEL_PATH, WILAYAH
from .teks import normalisasi, normalisasi_banyak


def buat_pipeline(C: float = 8.0) -> Pipeline:
    """Bangun pipeline TF-IDF (karakter + kata) dan Logistic Regression."""
    fitur = FeatureUnion(
        [
            (
                "karakter",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "kata",
                TfidfVectorizer(
                    analyzer="word",
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
        ],
        transformer_weights={"karakter": 1.0, "kata": 1.4},
    )

    return Pipeline(
        [
            ("normalisasi", FunctionTransformer(normalisasi_banyak, validate=False)),
            ("fitur", fitur),
            (
                "klasifier",
                LogisticRegression(
                    C=C,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


@dataclass
class DetektorBahasa:
    """Pembungkus pipeline beserta metadata model."""

    pipeline: Pipeline
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def kelas(self) -> list[str]:
        return list(self.pipeline.named_steps["klasifier"].classes_)

    def prediksi(self, teks: str, top_k: int = 3) -> dict[str, Any]:
        """Deteksi bahasa dari sebuah teks dan kembalikan hasil terstruktur."""
        if not isinstance(teks, str) or not teks.strip():
            raise ValueError("Teks tidak boleh kosong.")
        if len(teks) > 5000:
            raise ValueError("Teks terlalu panjang (maksimal 5000 karakter).")

        bersih = normalisasi(teks)
        if not bersih:
            raise ValueError("Teks tidak mengandung huruf yang bisa dianalisis.")

        kelas = self.kelas
        probabilitas = self._probabilitas(teks)
        urutan = probabilitas.argsort()[::-1][: max(1, top_k)]

        hasil_top = [
            {
                "bahasa": kelas[i],
                "keyakinan": round(float(probabilitas[i]), 6),
                "wilayah": WILAYAH.get(kelas[i], "-"),
            }
            for i in urutan
        ]
        teratas = hasil_top[0]

        kata_kunci = self.fitur_penting(teks, bahasa=teratas["bahasa"], top_n=6)

        peringatan: list[str] = []
        if len(bersih.split()) < 3:
            peringatan.append("Teks sangat pendek sehingga hasil bisa kurang akurat.")
        if teratas["keyakinan"] < MIN_CONFIDENCE:
            peringatan.append("Model belum cukup yakin; coba tambahkan kalimat lain.")
        elif len(hasil_top) > 1 and teratas["keyakinan"] - hasil_top[1]["keyakinan"] < 0.10:
            peringatan.append("Skor dua bahasa berdekatan; hasil perlu ditinjau manual.")

        return {
            "teks": teks.strip(),
            "teks_bersih": bersih,
            "bahasa": teratas["bahasa"],
            "keyakinan": teratas["keyakinan"],
            "wilayah": teratas["wilayah"],
            "top": hasil_top,
            "kata_kunci": kata_kunci,
            "peringatan": " ".join(peringatan) if peringatan else None,
        }

    def _probabilitas(self, teks: str):
        """Hitung probabilitas kelas; sediakan fallback softmax bila perlu."""
        klasifier = self.pipeline.named_steps["klasifier"]
        if hasattr(klasifier, "predict_proba"):
            return self.pipeline.predict_proba([teks])[0]

        skor = self.pipeline.decision_function([teks])[0]
        skor = skor - skor.max()
        eksp = np.exp(skor)
        return eksp / eksp.sum()

    def fitur_penting(
        self, teks: str, bahasa: str | None = None, top_n: int = 6
    ) -> list[str]:
        """Ambil kata/frasa yang paling mendorong prediksi sebuah bahasa."""
        if bahasa is None:
            bahasa = self.pipeline.predict([teks])[0]

        kelas = self.kelas
        if bahasa not in kelas:
            return []

        bersih = normalisasi(teks)
        fitur = self.pipeline.named_steps["fitur"]
        klasifier = self.pipeline.named_steps["klasifier"]
        koefisien = klasifier.coef_[kelas.index(bahasa)]

        # FeatureUnion menyusun fitur karakter lebih dulu, lalu fitur kata.
        offset = 0
        for nama, transformator in fitur.transformer_list:
            if nama == "kata":
                break
            offset += len(transformator.get_feature_names_out())

        vektor_kata = dict(fitur.transformer_list)["kata"]
        matriks = vektor_kata.transform([bersih])
        nama_fitur = vektor_kata.get_feature_names_out()

        skor: dict[str, float] = {}
        for indeks, nilai in zip(matriks.indices, matriks.data):
            kontribusi = float(koefisien[offset + indeks] * nilai)
            if kontribusi > 0:
                kata = str(nama_fitur[indeks])
                skor[kata] = max(skor.get(kata, 0.0), kontribusi)

        kandidat = [(k, s) for k, s in skor.items() if len(k.replace(" ", "")) >= 2]
        unigram = sorted(
            [(k, s) for k, s in kandidat if " " not in k],
            key=lambda item: item[1],
            reverse=True,
        )
        bigram = sorted(
            [(k, s) for k, s in kandidat if " " in k],
            key=lambda item: item[1],
            reverse=True,
        )

        hasil = [k for k, _ in unigram[:top_n]]
        if len(hasil) < min(3, top_n):
            hasil.extend(k for k, _ in bigram[: top_n - len(hasil)])
        if not hasil:
            hasil = sorted(set(bersih.split()), key=len, reverse=True)[:top_n]
        return hasil

    def simpan(self, path: str | Path = MODEL_PATH) -> Path:
        """Simpan pipeline dan metadata ke berkas joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metadata": self.metadata}, path, compress=3)
        return path

    @classmethod
    def dari_file(cls, path: str | Path = MODEL_PATH) -> "DetektorBahasa":
        """Muat model dari berkas joblib."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di {path}. Jalankan 'python train.py' terlebih dahulu."
            )
        isi = joblib.load(path)
        if isinstance(isi, dict) and "pipeline" in isi:
            return cls(pipeline=isi["pipeline"], metadata=isi.get("metadata", {}))
        return cls(pipeline=isi, metadata={})


def metadata_dasar(
    *,
    n_data: int,
    distribusi: dict[str, int],
    C: float,
    akurasi_holdout: float,
    macro_f1_holdout: float,
    akurasi_cv: float,
    macro_f1_cv: float,
) -> dict[str, Any]:
    """Susun metadata yang disimpan bersama model."""
    return {
        "nama_model": DESKRIPSI_MODEL,
        "dibuat": dt.datetime.now().isoformat(timespec="seconds"),
        "n_data": n_data,
        "jumlah_kelas": len(distribusi),
        "distribusi": distribusi,
        "parameter": {
            "C": C,
            "char_ngram": "2-5",
            "word_ngram": "1-2",
            "class_weight": "balanced",
        },
        "akurasi_holdout": round(float(akurasi_holdout), 6),
        "macro_f1_holdout": round(float(macro_f1_holdout), 6),
        "akurasi_cv": round(float(akurasi_cv), 6),
        "macro_f1_cv": round(float(macro_f1_cv), 6),
        "versi_sklearn": sklearn.__version__,
    }
