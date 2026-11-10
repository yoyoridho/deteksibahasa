const CONTOH = [
  {
    label: "Jawa",
    teks: "Sesuk aku arep tuku sepatu anyar ing mall cedhak omahku.",
  },
  {
    label: "Sunda",
    teks: "Isukan abdi badé mésér sapatu énggal di mall caket imah.",
  },
  {
    label: "Minang",
    teks: "Beko ambo nak mambali sapatu baru di mall dakek rumah.",
  },
  {
    label: "Batak",
    teks: "Marsogot ahu naeng manuhor sapatu imbaru di mall jonok jabu.",
  },
  {
    label: "Bugis",
    teks: "Esso ri uwamasei mallei sapatu baru ri mall macawa bola.",
  },
  {
    label: "Bali",
    teks: "Benjang titiang jagi meli sapatu anyar ring mall nampek umah.",
  },
  {
    label: "Madura",
    teks: "Laggu' sengko' arep mellé sapatu anyar e mall sempet romah.",
  },
  {
    label: "Banjar",
    teks: "Isuk ulun handak mambali sapatu hanyar di mall parak rumah.",
  },
  {
    label: "Indonesia",
    teks: "Besok saya akan membeli sepatu baru di mall dekat rumah.",
  },
];

const textarea = document.getElementById("teks");
const hitung = document.getElementById("hitung");
const tombolDeteksi = document.getElementById("deteksi");
const labelTombol = tombolDeteksi.querySelector(".btn-label");
const spinner = tombolDeteksi.querySelector(".spinner");
const kosong = document.getElementById("kosong");
const isi = document.getElementById("isi");
const errorBox = document.getElementById("error");
const peringatan = document.getElementById("peringatan");
const ring = document.getElementById("ring");

const micTombol = document.getElementById("mic");
const micLabel = micTombol.querySelector(".mic-label");
const micBahasa = document.getElementById("mic-bahasa");
const micStatus = document.getElementById("mic-status");
const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
let pengenalan = null;
let sedangMendengar = false;
let micAdaError = false;
let micAdaTeks = false;

const tombolTerjemah = document.getElementById("terjemahkan");
const labelTerjemah = tombolTerjemah.querySelector(".btn-label");
const spinnerTerjemah = tombolTerjemah.querySelector(".spinner");
const onlineCheck = document.getElementById("online");
const terjemahanHasil = document.getElementById("terjemahan-hasil");
const terjemahanTeks = document.getElementById("terjemahan-teks");
const terjemahanMetode = document.getElementById("terjemahan-metode");
const terjemahanCatatan = document.getElementById("terjemahan-catatan");
const terjemahanError = document.getElementById("terjemahan-error");

let deteksiTerakhir = null;

function formatPersen(nilai) {
  return (nilai * 100).toLocaleString("id-ID", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }) + "%";
}

function formatLebar(nilai) {
  return `${(nilai * 100).toFixed(3)}%`;
}

function perbaruiHitung() {
  const jumlah = textarea.value.length;
  hitung.textContent = `${jumlah.toLocaleString("id-ID")} / 5.000`;
}

function buatChip(teks, kelas = "chip") {
  const chip = document.createElement("span");
  chip.className = kelas;
  chip.textContent = teks;
  return chip;
}

function renderContoh() {
  const wadah = document.getElementById("contoh-list");
  CONTOH.forEach((contoh) => {
    const tombol = document.createElement("button");
    tombol.type = "button";
    tombol.className = "chip";
    tombol.textContent = contoh.label;
    tombol.addEventListener("click", () => {
      textarea.value = contoh.teks;
      perbaruiHitung();
      deteksi();
    });
    wadah.appendChild(tombol);
  });
}

function tampilkanError(pesan) {
  errorBox.textContent = pesan;
  errorBox.hidden = false;
}

function sembunyikanError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

const LABEL_METODE = {
  online: "Online · Google Translate",
  korpus: "Korpus paralel",
  kamus: "Kamus lokal",
  sama: "Sudah bahasa Indonesia",
  "tidak-tersedia": "Belum tersedia",
};

function resetTerjemahan() {
  terjemahanHasil.hidden = true;
  terjemahanTeks.textContent = "";
  terjemahanMetode.textContent = "";
  terjemahanMetode.removeAttribute("data-metode");
  terjemahanCatatan.textContent = "";
  terjemahanError.hidden = true;
  terjemahanError.textContent = "";
}

function renderTerjemahan(data) {
  terjemahanHasil.hidden = false;
  terjemahanTeks.textContent = data.terjemahan || "—";
  terjemahanMetode.textContent = LABEL_METODE[data.metode] || data.metode || "-";
  terjemahanMetode.dataset.metode = data.metode || "";
  terjemahanCatatan.textContent = data.catatan || "";
}

function tampilkanErrorTerjemahan(pesan) {
  terjemahanError.textContent = pesan;
  terjemahanError.hidden = false;
}

function aturMemuatTerjemahan(aktif) {
  tombolTerjemah.disabled = aktif;
  spinnerTerjemah.hidden = !aktif;
  labelTerjemah.textContent = aktif
    ? "Menerjemahkan..."
    : "Terjemahkan ke Indonesia";
}

async function terjemahkan() {
  const teks = textarea.value.trim();
  if (!teks) {
    tampilkanError("Masukkan teks terlebih dahulu.");
    textarea.focus();
    return;
  }

  const bahasa =
    deteksiTerakhir && deteksiTerakhir.teks === teks
      ? deteksiTerakhir.bahasa
      : null;

  aturMemuatTerjemahan(true);
  terjemahanError.hidden = true;

  try {
    const respons = await fetch("/api/terjemahkan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        teks,
        bahasa,
        online: onlineCheck.checked,
      }),
    });

    let data = {};
    try {
      data = await respons.json();
    } catch (_) {
      data = {};
    }

    if (!respons.ok) {
      throw new Error(data.error || `Permintaan gagal (HTTP ${respons.status}).`);
    }

    renderTerjemahan(data);
  } catch (kesalahan) {
    tampilkanErrorTerjemahan(
      kesalahan.message || "Terjadi kesalahan saat menerjemahkan."
    );
  } finally {
    aturMemuatTerjemahan(false);
  }
}

function aturStatusMic(pesan) {
  micStatus.textContent = pesan;
}

function aturMicAktif(aktif) {
  sedangMendengar = aktif;
  micTombol.classList.toggle("aktif", aktif);
  micTombol.setAttribute("aria-pressed", String(aktif));
  micLabel.textContent = aktif ? "Berhenti" : "Rekam suara";
  micBahasa.disabled = aktif;
}

function mulaiMikrofon() {
  if (!SpeechRecognitionAPI) {
    aturStatusMic(
      "Browser ini belum mendukung pengenalan suara. Gunakan Chrome atau Edge."
    );
    return;
  }

  if (sedangMendengar && pengenalan) {
    pengenalan.stop();
    return;
  }

  pengenalan = new SpeechRecognitionAPI();
  pengenalan.lang = micBahasa.value;
  pengenalan.continuous = true;
  pengenalan.interimResults = true;
  pengenalan.maxAlternatives = 1;

  micAdaError = false;
  micAdaTeks = false;
  pengenalan.onstart = () => {
    aturMicAktif(true);
    aturStatusMic("Mendengarkan… klik Berhenti jika sudah selesai.");
  };

  pengenalan.onresult = (event) => {
    let final = "";
    let sementara = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const transkrip = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        final += `${transkrip} `;
      } else {
        sementara += transkrip;
      }
    }

    if (final.trim()) {
      const dasar = textarea.value.trim();
      textarea.value = dasar ? `${dasar} ${final.trim()}` : final.trim();
      perbaruiHitung();
      resetTerjemahan();
      micAdaTeks = true;
    }

    aturStatusMic(
      sementara
        ? `Mendengar: ${sementara}`
        : "Mendengarkan… klik Berhenti jika sudah selesai."
    );
  };

  pengenalan.onerror = (event) => {
    micAdaError = true;
    const pesan = {
      "not-allowed": "Izin mic ditolak. Aktifkan izin mikrofon di browser.",
      "service-not-allowed": "Layanan pengenalan suara tidak diizinkan.",
      "no-speech": "Tidak ada suara yang terdeteksi. Coba lagi.",
      network: "Pengenalan suara membutuhkan koneksi internet.",
      "language-not-supported": "Bahasa ucapan ini belum didukung browser.",
    }[event.error];
    aturStatusMic(pesan || `Mic error: ${event.error}`);
  };

  pengenalan.onend = () => {
    aturMicAktif(false);
    if (!micAdaError) {
      aturStatusMic(
        micAdaTeks
          ? "Selesai. Teks hasil ucapan sudah ditambahkan."
          : "Tidak ada teks yang tertangkap. Coba lagi."
      );
    }
  };

  try {
    pengenalan.start();
  } catch (_) {
    aturMicAktif(false);
    aturStatusMic("Mic sedang sibuk. Tunggu sebentar lalu coba lagi.");
  }
}

function renderTopLain(daftar) {
  const wadah = document.getElementById("top-lain");
  wadah.replaceChildren();

  const lain = daftar.slice(1);
  if (lain.length === 0) {
    const li = document.createElement("li");
    li.className = "top-item";
    li.textContent = "Tidak ada kandidat lain.";
    wadah.appendChild(li);
    return;
  }

  lain.forEach((item) => {
    const li = document.createElement("li");
    li.className = "top-item";

    const nama = document.createElement("strong");
    nama.textContent = item.bahasa;

    const bar = document.createElement("div");
    bar.className = "mini-bar";
    const isiBar = document.createElement("span");
    bar.appendChild(isiBar);

    const skor = document.createElement("em");
    skor.textContent = formatPersen(item.keyakinan);

    li.append(nama, bar, skor);
    wadah.appendChild(li);
    isiBar.style.width = formatLebar(item.keyakinan);
  });
}

function renderHasil(hasil) {
  kosong.hidden = true;
  isi.hidden = false;
  sembunyikanError();
  resetTerjemahan();
  deteksiTerakhir = { teks: (hasil.teks || "").trim(), bahasa: hasil.bahasa };

  document.getElementById("bahasa").textContent = hasil.bahasa;
  document.getElementById("wilayah").textContent = hasil.wilayah || "-";
  document.getElementById("keyakinan").textContent = formatPersen(hasil.keyakinan);

  const derajat = Math.max(0, Math.min(360, hasil.keyakinan * 360));
  ring.style.setProperty("--nilai", `${derajat}deg`);

  const barUtama = document.getElementById("bar-utama");
  barUtama.style.width = formatLebar(hasil.keyakinan);

  if (hasil.peringatan) {
    peringatan.textContent = hasil.peringatan;
    peringatan.hidden = false;
  } else {
    peringatan.hidden = true;
    peringatan.textContent = "";
  }

  renderTopLain(hasil.top || []);

  const wadahKata = document.getElementById("kata-kunci");
  wadahKata.replaceChildren();
  (hasil.kata_kunci || []).forEach((kata) => {
    wadahKata.appendChild(buatChip(kata, "chip static"));
  });
  if (!hasil.kata_kunci || hasil.kata_kunci.length === 0) {
    wadahKata.appendChild(buatChip("—", "chip static"));
  }
}

function aturMemuat(aktif) {
  tombolDeteksi.disabled = aktif;
  spinner.hidden = !aktif;
  labelTombol.textContent = aktif ? "Menganalisis..." : "Deteksi Bahasa";
}

async function deteksi() {
  const teks = textarea.value.trim();
  if (!teks) {
    tampilkanError("Masukkan teks terlebih dahulu.");
    textarea.focus();
    return;
  }

  aturMemuat(true);
  sembunyikanError();

  try {
    const respons = await fetch("/api/deteksi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ teks }),
    });

    let data = {};
    try {
      data = await respons.json();
    } catch (_) {
      data = {};
    }

    if (!respons.ok) {
      throw new Error(data.error || `Permintaan gagal (HTTP ${respons.status}).`);
    }

    renderHasil(data);
  } catch (kesalahan) {
    tampilkanError(kesalahan.message || "Terjadi kesalahan saat menghubungi server.");
  } finally {
    aturMemuat(false);
  }
}

textarea.addEventListener("input", () => {
  perbaruiHitung();
  if (!terjemahanHasil.hidden) {
    resetTerjemahan();
  }
});
textarea.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    deteksi();
  }
});

document.getElementById("deteksi").addEventListener("click", deteksi);
document.getElementById("bersihkan").addEventListener("click", () => {
  textarea.value = "";
  perbaruiHitung();
  textarea.focus();
  sembunyikanError();
  resetTerjemahan();
  deteksiTerakhir = null;
  isi.hidden = true;
  kosong.hidden = false;
});
micTombol.addEventListener("click", mulaiMikrofon);
tombolTerjemah.addEventListener("click", terjemahkan);

if (!SpeechRecognitionAPI) {
  micTombol.disabled = true;
  aturStatusMic(
    "Mic tidak didukung browser ini. Gunakan Chrome atau Edge."
  );
} else if (!window.isSecureContext) {
  micTombol.disabled = true;
  aturStatusMic("Mic membutuhkan HTTPS atau localhost.");
}

document.getElementById("acak").addEventListener("click", () => {
  const contoh = CONTOH[Math.floor(Math.random() * CONTOH.length)];
  textarea.value = contoh.teks;
  perbaruiHitung();
  deteksi();
});

renderContoh();
perbaruiHitung();

// Dukungan tautan berbagi: /?teks=...&terjemahkan=1&online=0
const parameterAwal = new URLSearchParams(window.location.search);
const teksAwal = parameterAwal.get("teks");
const onlineAwal = parameterAwal.get("online");
if (onlineAwal === "0" || onlineAwal === "false") {
  onlineCheck.checked = false;
}
if (teksAwal) {
  textarea.value = teksAwal;
  perbaruiHitung();
  deteksi().then(() => {
    if (parameterAwal.get("terjemahkan") === "1" && deteksiTerakhir) {
      terjemahkan();
    }
  });
}
