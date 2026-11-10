"""Terjemahan bahasa daerah ke bahasa Indonesia (hybrid).

Urutan strategi:

1. **Online** — Google Translate untuk bahasa yang didukung.
2. **Korpus paralel** — cari kalimat sumber paling mirip, lalu ambil
   padanan bahasa Indonesianya.
3. **Kamus** — terjemahan kata/frasa per kata untuk kalimat yang belum ada
   di korpus.
4. Jika cakupan kamus terlalu rendah, kembalikan teks asli dengan catatan
   bahwa terjemahan belum tersedia.

Modul ini tidak membutuhkan koneksi internet saat ``online=False``.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    AMBANG_KORPUS,
    DATA_PATH,
    KAMUS_DIR,
    KODE_GOOGLE,
    MIN_CAKUPAN_KAMUS,
    MIN_IRISAN_KORPUS,
    MODEL_PATH,
    TERJEMAHAN_ONLINE,
    TERJEMAHAN_TIMEOUT,
)
from .dataset import muat_data
from .teks import normalisasi

# Kata = huruf Latin (termasuk aksen) dan angka, boleh memuat apostrof,
# termasuk apostrof di akhir kata (mis. "sengko'", "laggu'", "ta'").
_POLA_KATA = re.compile(
    r"([A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*['’]?)", re.UNICODE
)
_APOSTROF = str.maketrans({"’": "'", "‘": "'", "`": "'"})
_SPASI_GANDA = re.compile(r"[ \t]+")
_SPASI_SEBELUM_TANDA = re.compile(r"\s+([,.;:!?])")


def tanpa_aksen(teks: str) -> str:
    """Hilangkan aksen/diakritik tanpa mengubah huruf dasarnya."""
    return "".join(
        huruf
        for huruf in unicodedata.normalize("NFKD", teks)
        if not unicodedata.combining(huruf)
    )


def kunci_kata(kata: str) -> str:
    """Bentuk kunci kamus: huruf kecil, tanpa aksen, apostrof diseragamkan."""
    return tanpa_aksen(kata).lower().translate(_APOSTROF)


def _pecah_kata(teks: str) -> list[str]:
    """Pecah teks menjadi kata dan pemisah, tanpa membuang tanda baca.

    Hasilnya berselang-seling: pemisah, kata, pemisah, kata, ...
    Kata selalu berada pada indeks ganjil.
    """
    return _POLA_KATA.split(teks)


def _rapikan(teks: str) -> str:
    """Rapikan spasi ganda dan spasi sebelum tanda baca."""
    teks = _SPASI_GANDA.sub(" ", teks)
    teks = _SPASI_SEBELUM_TANDA.sub(r"\1", teks)
    return teks.strip()


def _muat_pasangan_paralel(path: str | Path) -> dict[str, list[tuple[str, str]]]:
    """Ambil pasangan (kalimat daerah, kalimat Indonesia) dari korpus."""
    teks, label, grup = muat_data(path)
    per_grup: dict[str, dict[str, str]] = defaultdict(dict)
    for kalimat, bahasa, nomor_grup in zip(teks, label, grup):
        per_grup[nomor_grup][bahasa] = kalimat

    pasangan: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for isi in per_grup.values():
        target = isi.get("Indonesia")
        if not target:
            continue
        for bahasa, sumber in isi.items():
            if bahasa != "Indonesia":
                pasangan[bahasa].append((sumber, target))
    return pasangan


class Penerjemah:
    """Mesin terjemahan hybrid bahasa daerah → bahasa Indonesia."""

    def __init__(
        self,
        kamus_dir: str | Path = KAMUS_DIR,
        korpus_path: str | Path = DATA_PATH,
        online: bool = TERJEMAHAN_ONLINE,
        timeout: float = TERJEMAHAN_TIMEOUT,
    ) -> None:
        self.online = bool(online)
        self.timeout = timeout
        self.kamus = self._muat_semua_kamus(Path(kamus_dir))
        self._frasa, self._maks_frasa = self._siapkan_frasa(self.kamus)
        self._korpus = self._siapkan_korpus(Path(korpus_path))

    # ------------------------------------------------------------------
    # Pemuatan data
    # ------------------------------------------------------------------
    @staticmethod
    def _muat_semua_kamus(kamus_dir: Path) -> dict[str, dict[str, str]]:
        kamus: dict[str, dict[str, str]] = {}
        if not kamus_dir.exists():
            return kamus
        for berkas in sorted(kamus_dir.glob("*.json")):
            try:
                isi = json.loads(berkas.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(isi, dict):
                kamus[berkas.stem] = {
                    str(k): str(v) for k, v in isi.items() if str(k).strip()
                }
        return kamus

    @staticmethod
    def _siapkan_frasa(
        kamus: dict[str, dict[str, str]],
    ) -> tuple[dict[str, dict[tuple[str, ...], str]], dict[str, int]]:
        """Ubah kamus menjadi lookup frasa berbasis kata yang sudah dinormalisasi."""
        lookup: dict[str, dict[tuple[str, ...], str]] = {}
        maks: dict[str, int] = {}
        for bahasa, isi in kamus.items():
            tabel: dict[tuple[str, ...], str] = {}
            for frasa, terjemahan in isi.items():
                kata = tuple(kunci_kata(bagian) for bagian in frasa.split())
                if kata and all(kata):
                    tabel[kata] = terjemahan
            lookup[bahasa] = tabel
            maks[bahasa] = max((len(k) for k in tabel), default=1)
        return lookup, maks

    @staticmethod
    def _siapkan_korpus(
        korpus_path: Path,
    ) -> dict[str, tuple[TfidfVectorizer, Any, list[str], list[str]]]:
        indeks: dict[str, tuple[TfidfVectorizer, Any, list[str], list[str]]] = {}
        if not korpus_path.exists():
            return indeks
        for bahasa, pasangan in _muat_pasangan_paralel(korpus_path).items():
            sumber = [normalisasi(kalimat) for kalimat, _ in pasangan]
            target = [kalimat for _, kalimat in pasangan]
            if not sumber:
                continue
            vektor = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
            )
            matriks = vektor.fit_transform(sumber)
            indeks[bahasa] = (vektor, matriks, sumber, target)
        return indeks

    # ------------------------------------------------------------------
    # Strategi terjemahan
    # ------------------------------------------------------------------
    def _terjemah_online(self, teks: str, kode: str) -> str:
        """Terjemahkan lewat endpoint publik Google Translate."""
        parameter = urllib.parse.urlencode(
            {"client": "gtx", "sl": kode, "tl": "id", "dt": "t", "q": teks}
        )
        url = "https://translate.googleapis.com/translate_a/single?" + parameter
        permintaan = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(permintaan, timeout=self.timeout) as respons:
            data = json.loads(respons.read().decode("utf-8"))

        bagian = data[0] if data else []
        hasil = "".join(
            potongan[0]
            for potongan in bagian
            if isinstance(potongan, list) and potongan and potongan[0]
        ).strip()
        if not hasil:
            raise ValueError("Google Translate mengembalikan hasil kosong.")
        return hasil

    def _cari_korpus(self, teks: str, bahasa: str) -> tuple[str, float, str] | None:
        """Cari kalimat padanan di korpus paralel dengan kemiripan karakter."""
        data = self._korpus.get(bahasa)
        if not data:
            return None
        vektor, matriks, sumber, target = data
        bersih = normalisasi(teks)
        if not bersih:
            return None
        kueri = vektor.transform([bersih])
        skor = cosine_similarity(kueri, matriks)[0]
        indeks = int(skor.argmax())
        nilai = float(skor[indeks])
        if nilai < AMBANG_KORPUS:
            return None

        # Kemiripan karakter saja bisa tinggi karena frasa yang sama, padahal
        # kalimat kueri jauh lebih pendek/berbeda. Periksa juga irisan kata.
        kata_kueri = set(tanpa_aksen(bersih).split())
        kata_sumber = set(tanpa_aksen(sumber[indeks]).split())
        gabungan = kata_kueri | kata_sumber
        irisan = len(kata_kueri & kata_sumber) / max(1, len(gabungan))
        if irisan < MIN_IRISAN_KORPUS:
            return None

        return target[indeks], nilai, sumber[indeks]

    def _terjemah_kamus(self, teks: str, bahasa: str) -> tuple[str, float]:
        """Terjemahkan kata/frasa dengan kamus; kembalikan (teks, cakupan)."""
        tabel = self._frasa.get(bahasa) or {}
        if not tabel:
            return teks, 0.0

        bagian = _pecah_kata(teks)
        posisi_kata = [i for i in range(1, len(bagian), 2)]
        if not posisi_kata:
            return teks, 0.0

        kunci = [kunci_kata(bagian[i]) for i in posisi_kata]
        maks = max(1, self._maks_frasa.get(bahasa, 1))
        terpakai: set[int] = set()
        jumlah_terjemah = 0

        for urutan, posisi in enumerate(posisi_kata):
            if posisi in terpakai:
                continue
            sisa = len(posisi_kata) - urutan
            for panjang in range(min(maks, sisa), 0, -1):
                frasa = tuple(kunci[urutan : urutan + panjang])
                if frasa not in tabel:
                    continue
                bagian[posisi] = tabel[frasa]
                for langkah in range(1, panjang):
                    posisi_lain = posisi_kata[urutan + langkah]
                    bagian[posisi_lain] = ""
                    terpakai.add(posisi_lain)
                terpakai.add(posisi)
                jumlah_terjemah += panjang
                break

        if jumlah_terjemah == 0:
            return teks, 0.0

        hasil = _rapikan("".join(bagian))
        if teks[:1].isupper() and hasil:
            hasil = hasil[:1].upper() + hasil[1:]
        cakupan = jumlah_terjemah / len(posisi_kata)
        return hasil, cakupan

    # ------------------------------------------------------------------
    # API utama
    # ------------------------------------------------------------------
    def terjemahkan(
        self,
        teks: str,
        bahasa: str | None = None,
        online: bool | None = None,
        detektor: Any | None = None,
    ) -> dict[str, Any]:
        """Terjemahkan teks ke bahasa Indonesia.

        Args:
            teks: Teks sumber.
            bahasa: Label bahasa sumber. Jika ``None``, bahasa dideteksi
                otomatis memakai model deteksi.
            online: ``True``/``False`` untuk memaksa mode online/offline.
                ``None`` mengikuti pengaturan instance.
            detektor: Instance ``DetektorBahasa`` opsional agar model tidak
                dimuat ulang saat deteksi otomatis.
        """
        if not isinstance(teks, str) or not teks.strip():
            raise ValueError("Teks tidak boleh kosong.")
        teks = teks.strip()
        if len(teks) > 5000:
            raise ValueError("Teks terlalu panjang (maksimal 5000 karakter).")

        if bahasa is None:
            if detektor is None:
                from .model import DetektorBahasa

                detektor = DetektorBahasa.dari_file(MODEL_PATH)
            bahasa = detektor.prediksi(teks)["bahasa"]

        if bahasa not in KODE_GOOGLE:
            raise ValueError(
                f"Bahasa tidak dikenal: {bahasa}. "
                f"Pilihan: {', '.join(KODE_GOOGLE)}."
            )

        if bahasa == "Indonesia":
            return self._hasil(
                teks,
                bahasa,
                teks,
                "sama",
                1.0,
                "Teks sudah dalam bahasa Indonesia.",
                False,
                None,
            )

        pakai_online = self.online if online is None else bool(online)
        kode = KODE_GOOGLE.get(bahasa)
        catatan_awal: str | None = None

        if pakai_online and kode:
            try:
                hasil_online = self._terjemah_online(teks, kode)
                return self._hasil(
                    teks,
                    bahasa,
                    hasil_online,
                    "online",
                    1.0,
                    "Diterjemahkan dengan Google Translate.",
                    True,
                    kode,
                )
            except Exception:
                catatan_awal = "Terjemahan online gagal; memakai mesin lokal."
        elif pakai_online and not kode:
            catatan_awal = (
                "Bahasa ini belum didukung Google Translate; memakai mesin lokal."
            )
        else:
            catatan_awal = "Mode offline: memakai korpus paralel dan kamus lokal."

        padanan = self._cari_korpus(teks, bahasa)
        if padanan is not None:
            hasil, skor, _sumber = padanan
            if skor >= 0.999:
                catatan = "Kalimat persis ditemukan di korpus paralel."
            else:
                catatan = (
                    "Kalimat padanan ditemukan di korpus paralel "
                    f"(kemiripan {skor:.0%})."
                )
            return self._hasil(
                teks,
                bahasa,
                hasil,
                "korpus",
                skor,
                self._gabung_catatan(catatan_awal, catatan),
                False,
                kode,
            )

        hasil_kamus, cakupan = self._terjemah_kamus(teks, bahasa)
        if cakupan >= MIN_CAKUPAN_KAMUS:
            catatan = (
                "Terjemahan kata per kata dari kamus "
                f"(cakupan {cakupan:.0%}); kualitas terbatas."
            )
            return self._hasil(
                teks,
                bahasa,
                hasil_kamus,
                "kamus",
                cakupan,
                self._gabung_catatan(catatan_awal, catatan),
                False,
                kode,
            )

        catatan = (
            "Terjemahan belum tersedia untuk kalimat ini. "
            "Coba aktifkan terjemahan online atau gunakan kalimat yang lebih umum."
        )
        return self._hasil(
            teks,
            bahasa,
            teks,
            "tidak-tersedia",
            cakupan,
            self._gabung_catatan(catatan_awal, catatan),
            False,
            kode,
        )

    @staticmethod
    def _gabung_catatan(*bagian: str | None) -> str:
        return " ".join(bagian for bagian in bagian if bagian)

    @staticmethod
    def _hasil(
        teks: str,
        bahasa: str,
        terjemahan: str,
        metode: str,
        kecocokan: float,
        catatan: str | None,
        online: bool,
        kode: str | None,
    ) -> dict[str, Any]:
        return {
            "teks": teks,
            "bahasa": bahasa,
            "terjemahan": terjemahan,
            "metode": metode,
            "kecocokan": round(float(kecocokan), 3) if kecocokan is not None else None,
            "catatan": catatan,
            "online": bool(online),
            "kode_bahasa": kode,
        }
