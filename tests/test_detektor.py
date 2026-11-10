"""Uji dasar untuk pipeline deteksi bahasa daerah.

Jalankan:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from bahasa_detector.config import DATA_PATH, MODEL_PATH
from bahasa_detector.dataset import muat_data, ringkasan_kelas
from bahasa_detector.teks import normalisasi


class TestNormalisasi(unittest.TestCase):
    def test_huruf_kecil_dan_tanda_baca(self):
        self.assertEqual(normalisasi("Aku, AREP!  Menyang..."), "aku arep menyang")

    def test_url_dan_mention(self):
        self.assertEqual(
            normalisasi("Cek https://contoh.id @budi #tag aku"), "cek aku"
        )

    def test_apostrof_dipertahankan(self):
        self.assertIn("ta'", normalisasi("Ta' ngakan"))

    def test_aksen_dipertahankan(self):
        self.assertIn("badé", normalisasi("Abdi badé angkat"))

    def test_input_bukan_string(self):
        with self.assertRaises(TypeError):
            normalisasi(None)  # type: ignore[arg-type]


class TestKorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.teks, cls.label, cls.grup = muat_data(DATA_PATH)

    def test_jumlah_data(self):
        self.assertEqual(len(self.teks), len(self.label))
        self.assertEqual(len(self.teks), len(self.grup))
        self.assertGreaterEqual(len(self.teks), 100)

    def test_sembilan_kelas(self):
        distribusi = ringkasan_kelas(self.label)
        self.assertGreaterEqual(len(distribusi), 9)
        self.assertTrue(all(jumlah >= 20 for jumlah in distribusi.values()))

    def test_grup_paralel(self):
        self.assertGreaterEqual(len(set(self.grup)), 45)


@unittest.skipUnless(MODEL_PATH.exists(), "Model belum dilatih; jalankan python train.py")
class TestPrediksi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bahasa_detector.model import DetektorBahasa

        cls.detektor = DetektorBahasa.dari_file(MODEL_PATH)

    def test_kalimat_yang_jelas(self):
        kasus = {
            "Aku arep menyang pasar tuku sayur kanggo masak sesuk.": "Jawa",
            "Abdi badé angkat ka pasar kanggo meuli sayuran.": "Sunda",
            "Ambo nak pai ka pasa mancari sayua.": "Minang",
            "Ahu naeng tu pasar manuhor eme.": "Batak",
            "Iyya lao ri pasa macari kaju.": "Bugis",
            "Titiang jagi lunga ka pasar meli jukut.": "Bali",
            "Sengko' arep mangkat ka pasar nyare sayur.": "Madura",
            "Ulun handak tulak ka pasar mambali sayur.": "Banjar",
            "Saya akan pergi ke pasar untuk membeli sayuran.": "Indonesia",
        }
        for teks, diharapkan in kasus.items():
            with self.subTest(bahasa=diharapkan):
                hasil = self.detektor.prediksi(teks)
                self.assertEqual(hasil["bahasa"], diharapkan)
                self.assertGreater(hasil["keyakinan"], 0.30)
                self.assertTrue(hasil["kata_kunci"])

    def test_kalimat_baru(self):
        kasus = {
            "Sesuk aku arep tuku sepatu anyar ing mall cedhak omahku.": "Jawa",
            "Isukan abdi badé mésér sapatu énggal di mall caket imah.": "Sunda",
            "Beko ambo nak mambali sapatu baru di mall dakek rumah.": "Minang",
            "Marsogot ahu naeng manuhor sapatu imbaru di mall jonok jabu.": "Batak",
            "Esso ri uwamasei mallei sapatu baru ri mall macawa bola.": "Bugis",
            "Benjang titiang jagi meli sapatu anyar ring mall nampek umah.": "Bali",
            "Laggu' sengko' arep mellé sapatu anyar e mall sempet romah.": "Madura",
            "Isuk ulun handak mambali sapatu hanyar di mall parak rumah.": "Banjar",
            "Besok saya akan membeli sepatu baru di mall dekat rumah.": "Indonesia",
        }
        for teks, diharapkan in kasus.items():
            with self.subTest(bahasa=diharapkan):
                hasil = self.detektor.prediksi(teks)
                self.assertEqual(hasil["bahasa"], diharapkan)

    def test_teks_kosong_ditolak(self):
        with self.assertRaises(ValueError):
            self.detektor.prediksi("   ")

    def test_hasil_memiliki_top_tiga(self):
        hasil = self.detektor.prediksi("Aku arep menyang pasar tuku sayur.")
        self.assertEqual(len(hasil["top"]), 3)
        self.assertAlmostEqual(hasil["top"][0]["keyakinan"], hasil["keyakinan"])


class TestTerjemahan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bahasa_detector.terjemah import Penerjemah

        cls.penerjemah = Penerjemah(online=False)

    def test_indonesia_tidak_diubah(self):
        hasil = self.penerjemah.terjemahkan(
            "Saya akan pergi ke pasar.", bahasa="Indonesia", online=False
        )
        self.assertEqual(hasil["metode"], "sama")
        self.assertEqual(hasil["terjemahan"], "Saya akan pergi ke pasar.")

    def test_korpus_paralel(self):
        hasil = self.penerjemah.terjemahkan(
            "Aku arep menyang pasar tuku sayur kanggo masak sesuk.",
            bahasa="Jawa",
            online=False,
        )
        self.assertEqual(hasil["metode"], "korpus")
        self.assertIn("pasar", hasil["terjemahan"].lower())

    def test_kamus_offline(self):
        hasil = self.penerjemah.terjemahkan(
            "Aku arep mangan.", bahasa="Jawa", online=False
        )
        self.assertEqual(hasil["metode"], "kamus")
        self.assertIn("makan", hasil["terjemahan"].lower())

    def test_aksen_dan_apostrof(self):
        sunda = self.penerjemah.terjemahkan(
            "Abdi badé mésér sapatu énggal.", bahasa="Sunda", online=False
        )
        self.assertIn("sepatu", sunda["terjemahan"].lower())
        madura = self.penerjemah.terjemahkan(
            "Laggu' sengko' arep melle sapatu.", bahasa="Madura", online=False
        )
        self.assertIn("sepatu", madura["terjemahan"].lower())
        self.assertIn("besok", madura["terjemahan"].lower())
        self.assertIn("saya", madura["terjemahan"].lower())

    def test_frasa_kamus_bugis(self):
        hasil = self.penerjemah.terjemahkan(
            "Esso ri uwamasei mallei sapatu baru.", bahasa="Bugis", online=False
        )
        self.assertIn("saya akan", hasil["terjemahan"].lower())
        self.assertIn("sepatu", hasil["terjemahan"].lower())

    def test_bugis_dan_banjar_tanpa_google(self):
        for teks, bahasa in [
            ("Iyya lao ri pasa macari jukut.", "Bugis"),
            ("Ulun handak mambali sayur hanyar.", "Banjar"),
        ]:
            with self.subTest(bahasa=bahasa):
                hasil = self.penerjemah.terjemahkan(
                    teks, bahasa=bahasa, online=False
                )
                self.assertIn(hasil["metode"], {"korpus", "kamus"})
                self.assertNotEqual(hasil["terjemahan"], teks)
                self.assertFalse(hasil["online"])

    def test_teks_kosong(self):
        with self.assertRaises(ValueError):
            self.penerjemah.terjemahkan("   ", bahasa="Jawa")

    def test_bahasa_tidak_dikenal(self):
        with self.assertRaises(ValueError):
            self.penerjemah.terjemahkan("Halo", bahasa="Aceh")


@unittest.skipUnless(MODEL_PATH.exists(), "Model belum dilatih; jalankan python train.py")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app

        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_beranda(self):
        respons = self.client.get("/")
        self.assertEqual(respons.status_code, 200)
        self.assertIn(b"Detektor", respons.data)

    def test_api_info(self):
        respons = self.client.get("/api/info")
        self.assertEqual(respons.status_code, 200)
        self.assertTrue(respons.get_json()["siap"])

    def test_api_deteksi(self):
        respons = self.client.post(
            "/api/deteksi", json={"teks": "Aku arep menyang pasar tuku sayur."}
        )
        self.assertEqual(respons.status_code, 200)
        data = respons.get_json()
        self.assertEqual(data["bahasa"], "Jawa")
        self.assertIn("top", data)

    def test_api_teks_kosong(self):
        respons = self.client.post("/api/deteksi", json={"teks": ""})
        self.assertEqual(respons.status_code, 400)
        self.assertIn("error", respons.get_json())

    def test_beranda_memiliki_mic_dan_terjemahan(self):
        respons = self.client.get("/")
        self.assertIn(b'id="mic"', respons.data)
        self.assertIn(b'id="terjemahkan"', respons.data)
        self.assertIn(b'id="online"', respons.data)

    def test_api_terjemahkan(self):
        respons = self.client.post(
            "/api/terjemahkan",
            json={
                "teks": "Aku arep menyang pasar tuku sayur kanggo masak sesuk.",
                "bahasa": "Jawa",
                "online": False,
            },
        )
        self.assertEqual(respons.status_code, 200)
        data = respons.get_json()
        self.assertEqual(data["bahasa"], "Jawa")
        self.assertIn(data["metode"], {"korpus", "kamus"})
        self.assertTrue(data["terjemahan"])

    def test_api_terjemahkan_auto_deteksi(self):
        respons = self.client.post(
            "/api/terjemahkan",
            json={
                "teks": "Ulun handak tulak ka pasar mambali sayur.",
                "online": False,
            },
        )
        self.assertEqual(respons.status_code, 200)
        data = respons.get_json()
        self.assertEqual(data["bahasa"], "Banjar")
        self.assertIn("pasar", data["terjemahan"].lower())

    def test_api_terjemahkan_teks_kosong(self):
        respons = self.client.post("/api/terjemahkan", json={"teks": ""})
        self.assertEqual(respons.status_code, 400)
        self.assertIn("error", respons.get_json())

    def test_api_terjemahkan_bahasa_tidak_dikenal(self):
        respons = self.client.post(
            "/api/terjemahkan",
            json={"teks": "Halo", "bahasa": "Aceh", "online": False},
        )
        self.assertEqual(respons.status_code, 400)
        self.assertIn("error", respons.get_json())


if __name__ == "__main__":
    unittest.main()
