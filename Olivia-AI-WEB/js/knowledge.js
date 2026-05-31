/**
 * ================================================================
 * OLIVIA AI — Knowledge Base v2.0 (knowledge.js)
 * Local RAG Engine untuk AIRA Customer Service
 *
 * Sistem: Keyword-Weighted Matching (No Embedding Required)
 * Scope: Permasalahan teknis game, akun, sistem, dan umum
 * Offline: 100% — tidak ada request API eksternal
 * ================================================================
 */

// ================================================================
// KNOWLEDGE BASE DATA
// Setiap entri memiliki: id, title, keywords[], text
// ================================================================

const KNOWLEDGE_BASE = [

  // ── PERFORMA & GRAFIS ────────────────────────────────────────
  {
    id: "perf-fps",
    title: "Mengatasi Lag dan FPS Rendah",
    keywords: [
      "lag", "fps", "lemot", "lambat", "berat", "macet", "patah",
      "stuttering", "jelag", "patahan", "grafis", "rendah", "frame",
      "drop", "terasa berat", "tidak lancar", "glitch visual"
    ],
    text: `Solusi Game Lag / FPS Rendah:\n1. Buka Pengaturan → Grafis, turunkan resolusi ke 720p.\n2. Nonaktifkan opsi 'Real-time Shadow' dan 'V-Sync'.\n3. Pastikan driver GPU Anda sudah versi terbaru (NVIDIA GeForce Experience / AMD Adrenalin).\n4. Tutup aplikasi berat di latar belakang: Chrome dengan banyak tab, Discord, OBS, Spotify.\n5. Pada laptop, aktifkan mode 'High Performance' di Power Options dan colokkan adaptor daya.\n6. Pastikan browser Anda versi terbaru dan aktifkan Hardware Acceleration di pengaturannya.`
  },

  {
    id: "perf-heat",
    title: "Masalah Panas Perangkat (Overheat)",
    keywords: [
      "panas", "overheat", "suhu tinggi", "throttle", "thermal",
      "gpu panas", "cpu panas", "kipas", "fan", "radiator", "berkeringat",
      "terlalu panas", "perangkat panas"
    ],
    text: `Jika perangkat terasa sangat panas saat bermain:\n1. Turunkan setting grafis ke 'Medium' atau 'Low' untuk mengurangi beban GPU.\n2. Pastikan ventilasi laptop tidak tersumbat — letakkan di permukaan keras dan datar.\n3. Gunakan laptop cooler (pendingin eksternal) untuk sirkulasi udara tambahan.\n4. Batasi sesi bermain maksimal 2 jam berturut-turut, lalu beri jeda 15 menit.\n5. Bersihkan debu di kipas laptop secara berkala (setiap 3-6 bulan) menggunakan semprotan udara.\n6. Cek suhu CPU/GPU dengan aplikasi HWMonitor — suhu aman di bawah 85°C saat gaming.`
  },

  {
    id: "perf-memory",
    title: "Masalah RAM dan Memori",
    keywords: [
      "ram", "memori", "out of memory", "kehabisan memori", "vram",
      "memory leak", "memori penuh", "crash memori", "not enough memory"
    ],
    text: `Jika game crash karena kehabisan memori:\n1. Tutup semua aplikasi lain sebelum memulai — terutama browser dengan banyak tab.\n2. Restart perangkat Anda sebelum sesi gaming untuk membersihkan memori.\n3. Pastikan RAM minimal 8 GB; 16 GB sangat direkomendasikan untuk performa optimal.\n4. Pada browser: buka hanya tab game.html saja, tutup semua tab lain.\n5. Jika menggunakan Windows, nonaktifkan startup apps yang tidak perlu via Task Manager.\n6. Model AI lokal membutuhkan ~2-4 GB RAM tersendiri — pastikan cukup tersedia.`
  },

  // ── AUDIO & SUARA ─────────────────────────────────────────────
  {
    id: "audio-no-sound",
    title: "Tidak Ada Suara atau Audio Bisu",
    keywords: [
      "suara", "audio", "bisu", "mute", "tidak ada suara",
      "speaker", "headphone", "volume", "silent", "sound off",
      "tidak terdengar", "audio hilang", "suara hilang"
    ],
    text: `Solusi Audio / Suara Tidak Terdengar:\n1. Periksa volume sistem di taskbar kanan bawah — pastikan tidak di-mute.\n2. Buka Volume Mixer (klik kanan ikon speaker → Open Volume Mixer) → pastikan volume browser tidak di-nol.\n3. Coba headphone atau speaker lain untuk mengkonfirmasi apakah masalah di perangkat atau game.\n4. Restart browser setelah menghubungkan perangkat audio baru.\n5. Pada Chrome/Edge: Settings → Privacy → Site Settings → cari olivia-ai → pastikan Sound diizinkan.\n6. Update atau reinstall driver audio melalui Device Manager jika masalah berlanjut.`
  },

  {
    id: "audio-echo",
    title: "Echo, Distorsi, atau Suara Aneh",
    keywords: [
      "echo", "gema", "distorsi", "feedback", "suara berulang",
      "suara aneh", "dengung", "noise", "kresek", "garbled",
      "suara kasar", "interferensi"
    ],
    text: `Jika ada echo, distorsi, atau interferensi suara:\n1. Jauhkan mikrofon dari speaker untuk mencegah audio feedback loop.\n2. Gunakan headset dengan mikrofon built-in — ini mencegah speaker menangkap output suara.\n3. Pada pengaturan audio Windows: klik kanan ikon speaker → Sound Settings → cek opsi 'Enhancements' → nonaktifkan efek yang tidak diperlukan.\n4. Pastikan tidak ada dua browser window yang membuka game secara bersamaan.\n5. Periksa kabel audio untuk kerusakan fisik atau koneksi yang longgar.`
  },

  // ── CRASH & FREEZE ────────────────────────────────────────────
  {
    id: "crash-freeze",
    title: "Game Crash atau Force Close Tiba-tiba",
    keywords: [
      "crash", "force close", "keluar sendiri", "tiba-tiba tutup",
      "hang", "freeze", "stuck", "tidak merespons", "berhenti",
      "error", "program berhenti", "aplikasi crash", "not responding"
    ],
    text: `Solusi Game Crash atau Force Close:\n1. Tutup semua aplikasi berat di latar belakang — terutama browser dengan banyak tab.\n2. Jalankan browser sebagai Administrator (klik kanan ikon browser → Run as Administrator).\n3. Hapus cache browser: Settings → Privacy → Clear browsing data → centang 'Cached images and files'.\n4. Update driver grafis ke versi paling baru via NVIDIA/AMD website resmi.\n5. Coba browser berbeda: Chrome atau Edge sangat direkomendasikan untuk WebGPU.\n6. Periksa apakah antivirus memblokir proses browser — tambahkan exception jika perlu.`
  },

  {
    id: "crash-loading",
    title: "Stuck di Loading atau Layar Hitam/Putih",
    keywords: [
      "loading", "loading lama", "layar hitam", "black screen",
      "white screen", "tidak muncul", "blank", "kosong",
      "loading terus", "spinning", "stuck loading", "tidak ada tampilan"
    ],
    text: `Jika game stuck di layar loading atau layar hitam:\n1. TUNGGU dulu — model AI berukuran besar (1-2 GB) memerlukan 1-5 menit loading pada pertama kali.\n2. Pastikan browser mendukung WebGPU: buka chrome://gpu di Chrome dan cari 'WebGPU'.\n3. Coba tekan F11 untuk toggle fullscreen — ini sering memicu render ulang yang berhasil.\n4. Bersihkan cache browser dan muat ulang halaman (Ctrl+Shift+R / Cmd+Shift+R).\n5. Buka konsol browser (F12 → Console) dan perhatikan pesan error merah untuk informasi lebih lanjut.\n6. Pastikan koneksi internet aktif jika ini adalah download pertama model AI.`
  },

  {
    id: "crash-webgpu",
    title: "Error WebGPU atau Model Tidak Bisa Dimuat",
    keywords: [
      "webgpu", "error webgpu", "gpu error", "shader", "tidak support webgpu",
      "model gagal", "failed to load", "model error", "wasm error",
      "onnx error", "transformers error", "ai error"
    ],
    text: `Jika terjadi error WebGPU atau model gagal dimuat:\n1. Pastikan menggunakan Chrome 113+ atau Edge 113+ — ini adalah persyaratan minimum WebGPU.\n2. Aktifkan Hardware Acceleration: Settings → System → aktifkan 'Use hardware acceleration when available'.\n3. Di Chrome: buka chrome://flags → cari 'WebGPU' → set ke 'Enabled'.\n4. Jika WebGPU tidak tersedia, sistem akan otomatis fallback ke CPU WASM (lebih lambat tapi tetap berfungsi).\n5. Restart browser setelah mengubah pengaturan flags.\n6. Pada perangkat tanpa GPU dedikasi, mode CPU WASM adalah satu-satunya opsi — respons lebih lambat namun tetap bekerja.`
  },

  // ── DOWNLOAD & INSTALASI ──────────────────────────────────────
  {
    id: "download-fail",
    title: "Gagal Download atau Update Model",
    keywords: [
      "download", "unduh", "update", "patch", "install", "instalasi",
      "corrupted", "gagal download", "terputus", "lambat download",
      "download stuck", "resume download", "download error", "timeout"
    ],
    text: `Solusi Gagal Download atau Update:\n1. Pastikan koneksi internet stabil dengan kecepatan minimal 5 Mbps — model AI berukuran ~1.2 GB.\n2. Jangan tutup browser selama proses download berlangsung.\n3. Pastikan ruang penyimpanan browser minimal 5 GB tersedia (cek IndexedDB storage).\n4. Jika download terputus di tengah: muat ulang halaman — Transformers.js mendukung resume otomatis.\n5. Coba matikan sementara VPN atau proxy yang mungkin membatasi koneksi ke HuggingFace.\n6. Jika dari jaringan kampus/kantor, hubungi admin IT — beberapa firewall memblokir CDN HuggingFace.`
  },

  // ── DATA & SAVE ───────────────────────────────────────────────
  {
    id: "data-save",
    title: "Masalah Save Data, Progress, atau Riwayat Chat",
    keywords: [
      "save", "menyimpan", "data hilang", "progress hilang", "load",
      "backup", "reset data", "tersimpan", "chat hilang", "history",
      "riwayat", "percakapan hilang", "localstorage"
    ],
    text: `Tentang Sistem Penyimpanan Data Olivia AI:\n1. Seluruh data chat tersimpan di localStorage browser Anda secara otomatis.\n2. PENTING: Jangan gunakan 'Clear Site Data' atau 'Clear All Cookies and Cache' — riwayat chat akan terhapus permanen.\n3. Untuk backup: gunakan fitur Export via menu Settings (jika tersedia) atau salin manual dari LocalStorage.\n4. Data bersifat lokal — tidak ada sinkronisasi cloud. Berganti browser atau device berarti data tidak terbawa.\n5. Untuk melihat data tersimpan: buka DevTools (F12) → Application → Local Storage → cari 'olivia_chat_*'.\n6. Jika ingin reset percakapan: gunakan tombol 'Hapus' di panel chat, bukan clear browser data.`
  },

  // ── CONTROLLER & INPUT ────────────────────────────────────────
  {
    id: "input-controller",
    title: "Masalah Controller atau Input Device",
    keywords: [
      "controller", "gamepad", "joystick", "tidak terdeteksi",
      "keyboard", "mouse", "tombol tidak berfungsi", "input error",
      "gamepad api", "vibration", "rumble"
    ],
    text: `Jika controller atau input device tidak terdeteksi:\n1. Hubungkan controller via USB untuk koneksi yang lebih stabil dibanding Bluetooth.\n2. Buka Device Manager (Win+X → Device Manager) → cari controller → klik kanan → Update driver.\n3. Test di Windows Game Controller: tekan Win+R → ketik 'joy.cpl' → periksa status controller.\n4. Pada Chrome/Edge: pastikan browser memiliki izin akses Gamepad API (biasanya otomatis).\n5. Restart game setelah menghubungkan controller — browser perlu mendeteksi ulang perangkat input.\n6. Jika menggunakan controller Xbox via Bluetooth: pastikan Xbox Wireless Adapter atau Bluetooth tersambung aktif.`
  },

  // ── LOGIN & AKUN ──────────────────────────────────────────────
  {
    id: "account-login",
    title: "Masalah Login dan Manajemen Akun",
    keywords: [
      "login", "masuk", "account", "akun", "username", "nama",
      "lupa nama", "ganti nama", "tidak bisa masuk", "login error",
      "session", "sesi", "keluar", "logout"
    ],
    text: `Sistem Akun Lokal Olivia AI:\n1. Demo web menggunakan sistem nama lokal — tidak diperlukan email atau password.\n2. Nama Anda tersimpan di localStorage browser (bukan di server kami).\n3. Untuk mengganti nama: klik tombol 'Keluar' di sudut kiri atas → masukkan nama baru → masuk kembali.\n4. Setiap nama memiliki riwayat chat terpisah — berganti nama akan memulai percakapan baru.\n5. Data akun tidak tersinkronisasi antar-device secara otomatis (fitur cloud sync belum tersedia).\n6. Untuk reset total akun: Settings Browser → Site Settings → Clear Data untuk domain olivia-ai.`
  },

  // ── BROWSER & WEBGPU COMPATIBILITY ───────────────────────────
  {
    id: "browser-compat",
    title: "Kompatibilitas Browser dan Persyaratan WebGPU",
    keywords: [
      "browser", "chrome", "firefox", "edge", "safari", "webgpu",
      "tidak support", "tidak kompatibel", "versi lama", "browser lama",
      "chromium", "opera", "brave", "arc browser"
    ],
    text: `Persyaratan Browser untuk Olivia AI WebGPU:\n1. TERBAIK: Google Chrome 113+ atau Microsoft Edge 113+ → performa WebGPU optimal.\n2. Brave Browser (berbasis Chromium 113+) juga didukung dengan performa setara.\n3. Firefox: perlu mengaktifkan WebGPU secara manual di about:config → dom.webgpu.enabled → true.\n4. Safari: dukungan WebGPU sangat terbatas; gunakan Chrome/Edge untuk pengalaman terbaik.\n5. Aktifkan Hardware Acceleration: Settings → System → 'Use hardware acceleration when available' → Restart.\n6. Untuk verifikasi WebGPU aktif: buka chrome://gpu → cari 'WebGPU' di bagian 'Graphics Feature Status'.`
  },

  // ── SPESIFIKASI SISTEM ────────────────────────────────────────
  {
    id: "sys-requirements",
    title: "Spesifikasi Sistem Minimum dan Rekomendasi",
    keywords: [
      "spesifikasi", "spec", "requirements", "minimum", "recommended",
      "sistem operasi", "os", "windows", "macos", "linux",
      "ram", "gpu", "vram", "cpu", "processor", "hardware"
    ],
    text: `Spesifikasi Sistem untuk Olivia AI:\n\nMINIMUM:\n- OS: Windows 10 64-bit / macOS 12 Monterey / Ubuntu 20.04\n- CPU: Intel Core i5 Gen-8 atau AMD Ryzen 5 3000 series\n- RAM: 8 GB\n- GPU: Integrated Graphics (menggunakan mode CPU WASM — lebih lambat)\n- Storage: 5 GB free space untuk cache model\n- Browser: Chrome 113+ atau Edge 113+\n\nDIREKOMENDASIKAN:\n- CPU: Intel Core i7 Gen-10+ atau AMD Ryzen 7 5000+\n- RAM: 16 GB+\n- GPU: NVIDIA GTX 1060 6GB+ atau AMD RX 580 8GB+ (untuk WebGPU penuh)\n- Storage: 10 GB SSD free space\n- Internet: Hanya untuk download model awal ~1.2 GB`
  },

  // ── MOBILE & ANDROID ─────────────────────────────────────────
  {
    id: "mobile-android",
    title: "Penggunaan di Perangkat Mobile atau Android",
    keywords: [
      "hp", "android", "ios", "mobile", "ponsel", "tablet",
      "smartphone", "iphone", "aplikasi android", "apk",
      "play store", "app store", "mobile version"
    ],
    text: `Tentang Olivia AI di Perangkat Mobile:\n1. Versi web dapat diakses via browser Chrome di Android — namun performa terbatas pada HP mid-range.\n2. Dukungan WebGPU di Chrome Android (versi 121+) masih dalam tahap eksperimental.\n3. Untuk HP high-end (Snapdragon 8 Gen 2+): aktifkan WebGPU via chrome://flags di Chrome Android.\n4. Versi aplikasi Android native (via CapacitorJS + MediaPipe) sedang dalam pengembangan untuk performa NPU optimal.\n5. Sementara itu, mode CPU WASM berfungsi di hampir semua HP Android modern dengan RAM 6 GB+.\n6. iPad dengan chip M1/M2 memiliki performa WebGPU yang sangat baik via Chrome iOS.`
  },

  // ── PRIVASI & KEAMANAN ────────────────────────────────────────
  {
    id: "privacy-security",
    title: "Privasi Data dan Keamanan",
    keywords: [
      "privasi", "privacy", "keamanan", "data", "apakah aman",
      "tracking", "server", "cloud", "upload data", "rekam", "disimpan dimana",
      "rahasia", "aman", "secure"
    ],
    text: `Jaminan Privasi Olivia AI:\n1. TIDAK ADA data yang dikirim ke server eksternal — seluruh komputasi terjadi di browser Anda.\n2. Model AI berjalan sepenuhnya lokal — percakapan tidak pernah meninggalkan perangkat Anda.\n3. Riwayat chat tersimpan di localStorage browser Anda — hanya Anda yang bisa mengaksesnya(untuk memastikan sepenuhnya offline tanpa internet harap unduh versi game, hanya khusus PC!).\n4. Olivia AI tidak memiliki akun user di server, tidak ada database eksternal, tidak ada analytics tracking.\n5. Untuk memastikan tidak ada data yang keluar: Anda bisa memutus internet setelah model berhasil dimuat(berlaku untuk versi game).\n6. Source code proyek ini bersifat terbuka untuk diaudit — transparansi penuh adalah prioritas kami.`
  },

  // ── TIPS & PANDUAN ────────────────────────────────────────────
  {
    id: "tips-general",
    title: "Tips Optimal dan Panduan Bermain",
    keywords: [
      "tips", "bantuan", "help", "panduan", "tutorial", "cara", "saran",
      "rekomendasi", "bagaimana", "gimana", "petunjuk", "guide",
      "mulai", "getting started", "pertama kali"
    ],
    text: `Tips Bermain Olivia AI Secara Optimal:\n1. Pertama kali: tunggu proses download model AI selesai (~1-5 menit tergantung kecepatan internet).\n2. Setelah model terunduh sekali, game berjalan sepenuhnya OFFLINE tanpa internet.\n3. Gunakan headphone untuk pengalaman audio paling imersif.\n4. Tab 'Chat Olivia' untuk percakapan santai; Tab 'AIRA Support' untuk bantuan teknis.\n5. Riwayat chat tersimpan otomatis — tidak perlu menekan tombol save manual.\n6. Gunakan shortcut: Ctrl+1 untuk Chat Olivia, Ctrl+2 untuk AIRA Support.\n7. Jika Olivia lambat merespons, itu wajar — model 2B parameter membutuhkan beberapa detik.`
  },

  // ── ERROR SPESIFIK ────────────────────────────────────────────
  {
    id: "error-specific",
    title: "Pesan Error Spesifik dan Artinya",
    keywords: [
      "error message", "pesan error", "kode error", "404", "500",
      "undefined", "null", "failed to fetch", "cors", "network error",
      "module error", "import error", "script error"
    ],
    text: `Panduan Membaca Pesan Error Olivia AI:\n- "Failed to fetch model" → Masalah koneksi internet saat download model. Periksa koneksi dan coba lagi.\n- "WebGPU not supported" → Browser tidak mendukung WebGPU. Gunakan Chrome/Edge terbaru atau aktifkan di flags. Jika tidak berhasil maka unduh versi game-nya.\n- "Out of memory" → RAM perangkat tidak mencukupi. Tutup semua aplikasi lain dan coba lagi.\n- "CORS error" → Jangan buka file HTML langsung (file:// protocol) — gunakan local server atau hosting.\n- "Module not found" → File JS tidak ditemukan. Pastikan struktur folder proyek lengkap dan benar.\n- Untuk error lain: buka DevTools F12 → Console → salin pesan error dan hubungi support kami. \n- Untuk beberapa alasan, Panduan kurang membantu, jika tidak memungkinkan jalan dengan WebGPU karena masalah teknis. Harap unduh versi game(khusus PC, untuk Android sedang dalam tahap pengembangan).`
  }

];

// ================================================================
// RAG SEARCH ENGINE — Keyword-Weighted Matching
// ================================================================

/**
 * Cari entri pengetahuan yang paling relevan berdasarkan query user.
 * Menggunakan sistem skor bobot untuk relevansi optimal.
 *
 * @param {string} query   - Pertanyaan dari user (teks bebas)
 * @param {number} [topK=2] - Jumlah artikel teratas yang dikembalikan
 * @returns {string}       - Teks gabungan dari artikel paling relevan, atau "" jika tidak ada
 */
export function cariPengetahuanLokal(query, topK = 2) {
  if (!query || typeof query !== "string") return "";

  const queryLower = query.toLowerCase().trim();
  if (queryLower.length < 2) return "";

  const hasil = [];

  for (const item of KNOWLEDGE_BASE) {
    const skor = hitungSkorRelevansi(queryLower, item.keywords, item.title);
    if (skor > 0) {
      hasil.push({ skor, konten: `[${item.title}]\n${item.text}` });
    }
  }

  if (hasil.length === 0) return "";

  // Urutkan berdasarkan skor tertinggi, ambil top K
  return hasil
    .sort((a, b) => b.skor - a.skor)
    .slice(0, topK)
    .map((item) => item.konten)
    .join("\n\n---\n\n");
}

/**
 * Hitung skor relevansi antara query dan satu entri knowledge base.
 *
 * Sistem bobot:
 *   +3 untuk setiap keyword yang persis ada di query
 *   +1 untuk partial match (keyword 4+ karakter cocok di awal kata query)
 *   +4 bonus jika kata dari judul artikel cocok dengan query (kata > 4 huruf)
 *
 * @param {string}   query    - Query user (sudah lowercase)
 * @param {string[]} keywords - Daftar keyword artikel
 * @param {string}   title    - Judul artikel
 * @returns {number}          - Skor relevansi (0 = tidak relevan)
 */
function hitungSkorRelevansi(query, keywords, title) {
  let skor = 0;

  // Tokenisasi query menjadi kata-kata individual
  const kataQuery = query.split(/\s+/);

  for (const kw of keywords) {
    const kwLower = kw.toLowerCase();

    if (query.includes(kwLower)) {
      // Exact match — bobot tertinggi
      skor += 3;
    } else if (kwLower.length >= 4) {
      // Partial match: awalan 4 karakter keyword cocok di salah satu kata query
      const awalan = kwLower.slice(0, 4);
      if (kataQuery.some((kata) => kata.startsWith(awalan))) {
        skor += 1;
      }
    }
  }

  // Bonus title match: kata panjang dari judul yang ada di query
  const kataJudul = title.toLowerCase().split(/\s+/);
  for (const kata of kataJudul) {
    if (kata.length > 4 && query.includes(kata)) {
      skor += 4;
      break; // Hanya satu bonus title per artikel
    }
  }

  return skor;
}

// ================================================================
// UTILITY FUNCTIONS
// ================================================================

/**
 * Dapatkan daftar semua topik yang tersedia (untuk help menu / dokumentasi)
 * @returns {{ id: string, title: string, keywordCount: number }[]}
 */
export function dapatkanDaftarTopik() {
  return KNOWLEDGE_BASE.map(({ id, title, keywords }) => ({
    id,
    title,
    keywordCount: keywords.length,
  }));
}

/**
 * Cari satu entri spesifik berdasarkan ID
 * @param {string} id - ID entri
 * @returns {object|null}
 */
export function cariEntriById(id) {
  return KNOWLEDGE_BASE.find((item) => item.id === id) ?? null;
}

/**
 * Dapatkan jumlah total entri dalam knowledge base
 * @returns {number}
 */
export function jumlahEntri() {
  return KNOWLEDGE_BASE.length;
}

/**
 * Dapatkan semua keyword unik dari seluruh knowledge base
 * @returns {string[]}
 */
export function semuaKeyword() {
  const set = new Set();
  for (const item of KNOWLEDGE_BASE) {
    for (const kw of item.keywords) {
      set.add(kw.toLowerCase());
    }
  }
  return Array.from(set).sort();
}
