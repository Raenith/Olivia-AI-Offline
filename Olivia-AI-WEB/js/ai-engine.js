/**
 * ================================================================
 * OLIVIA AI — AI Engine v4.0 (ai-engine.js)
 *
 * Features:
 *   - Transformers.js v3 via esm.sh (Client-Side Edge AI)
 *   - WebGPU acceleration (mandatory — no WASM fallback)
 *   - Gemma4ForConditionalGeneration + AutoProcessor (terpisah)
 *   - TextStreamer: token streaming real-time via callback
 *   - Multi-Agent: Olivia (companion) + AIRA ACS (customer service)
 *   - Emotion Engine: deteksi sentimen → injeksi perilaku dinamis
 *   - Local RAG: keyword-matching dari knowledge.js
 *   - Multi-turn memory via localStorage (keyed by username+mode)
 *   - Token window management: slice(-MAX_WINDOW)
 *
 * Model: ACS (lokal — ./assets/models/ACS/)
 * Loader: Gemma4ForConditionalGeneration + AutoProcessor
 * Storage: localStorage browser (100% offline, privat)
 * Offline: Ya — model diambil dari path lokal, tidak butuh internet
 * ================================================================
 */

// ================================================================
// STATIC IMPORT — Transformers.js v3 via esm.sh
// ================================================================

import {
  env,
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  TextStreamer,
} from 'https://esm.sh/@huggingface/transformers';

import { cariPengetahuanLokal } from './knowledge.js';

// ================================================================
// ENVIRONMENT CONFIGURATION — Local Model Only
// ================================================================

/**
 * Paksa Transformers.js untuk hanya menggunakan model lokal.
 * Semua fetch ke HuggingFace Hub dinonaktifkan.
 * Model dibaca langsung dari path lokal yang ditentukan.
 */
env.allowRemoteModels = false;
env.allowLocalModels  = true;
env.localModelPath    = './assets/models/';

// ================================================================
// CONSTANTS
// ================================================================

/**
 * ID model lokal — direktori: ./assets/models/
 * Transformers.js akan me-resolve ke: env.localModelPath + MODEL_ID
 */
const MODEL_ID = 'ACS';

/**
 * Jumlah token baru maksimum yang di-generate per respons.
 * Olivia: 100 (percakapan singkat & natural)
 * AIRA:   180 (perlu ruang untuk numbered steps)
 */
const MAX_NEW_TOKENS_OLIVIA = 100;
const MAX_NEW_TOKENS_AIRA   = 180;

/**
 * Jumlah pesan terakhir yang dimasukkan ke context window.
 * 8 pesan = 4 giliran (4 user + 4 AI).
 * Lebih dari ini akan menyebabkan token overflow pada model kecil.
 */
const MAX_CONTEXT_MESSAGES = 8;

/**
 * Stop sequences — model akan berhenti generate saat bertemu string ini.
 */
const STOP_SEQUENCES_OLIVIA = ['User:', 'Kamu:', '<|eot_id|>', '<|end_of_text|>', '[INST]'];
const STOP_SEQUENCES_AIRA   = ['User:', 'AIRA:', '<|eot_id|>', '<|end_of_text|>', '[INST]'];

// ================================================================
// EMOTION ENGINE — Keyword Dictionary
// ================================================================

/**
 * Daftar kata kunci untuk deteksi sentimen teks input user.
 * Digunakan untuk memodifikasi persona Olivia secara dinamis.
 * JANGAN KURANGI — setiap kosakata berpengaruh pada akurasi scoring.
 */
const EMOSI_SEDIH = [
  'sedih', 'nangis', 'menangis', 'air mata', 'kesepian', 'sendirian',
  'galau', 'patah hati', 'putus', 'ditinggal', 'ditinggalkan',
  'kecewa', 'menyesal', 'nyesel', 'gagal', 'hancur', 'sakit hati',
  'depresi', 'hopeless', 'hopeles', 'down', 'brokenheart', 'duka',
  'kehilangan', 'hilang semangat', 'tidak semangat', 'capek hidup',
  'lelah', 'exhausted', 'hampa', 'kosong', 'tak berharga'
];

const EMOSI_MARAH = [
  'marah', 'kesal', 'frustrasi', 'frustasi', 'benci', 'sialan',
  'bodoh', 'capek', 'muak', 'sebel', 'dongkol', 'gondok',
  'annoyed', 'pissed', 'angry', 'emosi', 'ngamuk', 'jengkel',
  'gak tahan', 'capek banget', 'bosan', 'boring', 'paling sebel'
];

const EMOSI_BAHAGIA = [
  'senang', 'bahagia', 'gembira', 'suka', 'cinta', 'sayang',
  'kangen', 'rindu', 'happy', 'excited', 'semangat', 'bangga',
  'berhasil', 'sukses', 'lulus', 'menang', 'dapat', 'dapet',
  'amazing', 'keren', 'asik', 'asyik', 'seru', 'fun', 'wow'
];

const EMOSI_CEMAS = [
  'takut', 'khawatir', 'cemas', 'anxiety', 'panik', 'panic',
  'nervous', 'deg-degan', 'was-was', 'gelisah', 'gugup',
  'tidak yakin', 'ragu', 'bingung', 'confused', 'insecure'
];

// ================================================================
// MODULE STATE (Singleton)
// ================================================================

/** Instance AutoProcessor — null sebelum diinisialisasi */
let processor = null;

/** Instance Gemma4ForConditionalGeneration — null sebelum diinisialisasi */
let model = null;

/** Flag loading untuk mencegah double-init */
let _isLoading = false;

// ================================================================
// PUBLIC: AI INITIALIZATION
// ================================================================

/**
 * Inisialisasi AI Engine v4.0 (singleton).
 * Memuat AutoProcessor dan Gemma4ForConditionalGeneration secara terpisah
 * dari path model lokal. Memerlukan WebGPU — tidak ada fallback CPU.
 *
 * @param {(status: string) => void} onProgress - Callback update teks status ke UI
 * @returns {Promise<void>}
 * @throws {Error} Jika WebGPU tidak tersedia atau model gagal dimuat
 */
export async function loadAIEngine(onProgress = () => {}) {
  // Jika sudah siap, langsung return
  if (processor !== null && model !== null) {
    onProgress('🟢 AI Engine v4.0 aktif — 100% Offline (Local Model)');
    return;
  }

  // Jika sedang dalam proses loading, tunggu selesai
  if (_isLoading) {
    onProgress('⏳ Menunggu proses loading sebelumnya...');
    await _waitForLoad();
    return;
  }

  _isLoading = true;

  try {
    onProgress('🔍 Memeriksa dukungan WebGPU...');

    // Verifikasi WebGPU tersedia — v4.0 memerlukan GPU, tidak ada WASM fallback
    const webgpuTersedia = await _cekWebGPU();
    if (!webgpuTersedia) {
      throw new Error('WebGPU tidak tersedia. Gunakan Chrome/Edge terbaru dan aktifkan Hardware Acceleration di pengaturan browser.');
    }
    onProgress('✅ WebGPU terdeteksi — menggunakan akselerasi GPU penuh.');

    // --- Muat AutoProcessor ---
    onProgress(`📦 Memuat AutoProcessor dari model lokal "${MODEL_ID}"...`);
    processor = await AutoProcessor.from_pretrained(MODEL_ID, {
      progress_callback: (info) => {
        if (info.status === 'progress') {
          onProgress(`📦 Memuat Processor... ${Math.round(info.progress)}%`);
        } else if (info.status === 'ready') {
          onProgress('✅ AutoProcessor siap!');
        }
      }
    });
    onProgress('🟢 AutoProcessor berhasil dimuat.');

    // --- Muat Gemma4ForConditionalGeneration ---
    onProgress(`⚙️ Memuat model Gemma4 "${MODEL_ID}" ke WebGPU (q4)...`);
    onProgress('💡 Model dibaca dari penyimpanan lokal — tidak memerlukan internet.');

    model = await Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype:  'q4',
      device: 'webgpu',
      progress_callback: (info) => {
        if (info.status === 'progress') {
          onProgress(`⚙️ Memuat file model... ${Math.round(info.progress)}%`);
        } else if (info.status === 'ready') {
          onProgress('🟢 Model Gemma4 siap di GPU!');
        }
      }
    });

    onProgress('🟢 AI Engine v4.0 aktif — Siap 100% Offline! (Local WebGPU)');

  } catch (err) {
    // Reset state agar retry bisa dilakukan
    _isLoading = false;
    processor  = null;
    model      = null;
    console.error('[AI Engine v4.0] Gagal inisialisasi:', err);

    // Pesan error yang informatif dan ramah pengguna
    const msg = String(err?.message ?? err);
    if (msg.includes('WebGPU') || msg.includes('GPU')) {
      throw new Error('WebGPU tidak tersedia atau tidak didukung. Gunakan Chrome/Edge versi terbaru dan aktifkan Hardware Acceleration di pengaturan browser.');
    } else if (msg.includes('fetch') || msg.includes('network') || msg.includes('404')) {
      throw new Error(`Model lokal tidak ditemukan di path "${env.localModelPath}${MODEL_ID}". Pastikan direktori model sudah tersedia.`);
    } else if (msg.includes('memory') || msg.includes('OOM') || msg.includes('out of memory')) {
      throw new Error('Memori GPU tidak mencukupi. Tutup semua tab dan aplikasi lain, lalu coba lagi.');
    } else {
      throw new Error(`Gagal memuat AI Engine v4.0: ${msg}`);
    }
  } finally {
    _isLoading = false;
  }
}

/**
 * Cek apakah AI Engine sudah siap digunakan.
 * @returns {boolean}
 */
export function isAIReady() {
  return processor !== null && model !== null;
}

// ================================================================
// PUBLIC: CHAT HISTORY (localStorage)
// ================================================================

/**
 * Generate storage key yang unik berdasarkan username dan mode.
 * Format: olivia_chat_{username}_{mode}
 */
function _storageKey(username, mode) {
  // Sanitasi: hanya huruf, angka, underscore, dan strip karakter berbahaya
  const safeName = String(username).replace(/[^a-zA-Z0-9_\-ก-๙ก-ฮ]/g, '_').slice(0, 32);
  const safeMode = String(mode).replace(/[^a-zA-Z0-9_]/g, '_');
  return `olivia_chat_${safeName}_${safeMode}`;
}

/**
 * Ambil seluruh riwayat chat dari localStorage.
 *
 * @param {string} username
 * @param {string} mode - "olivia" | "aira_acs"
 * @returns {{ role: "user"|"assistant", message: string, time: string }[]}
 */
export function dapatkanRiwayatChat(username, mode) {
  if (!username || !mode) return [];
  try {
    const raw = localStorage.getItem(_storageKey(username, mode));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * Simpan satu pesan baru ke riwayat chat di localStorage.
 * Jika storage penuh (QuotaExceededError), pangkas 20 pesan terlama dan coba lagi.
 *
 * @param {string}             username
 * @param {string}             mode
 * @param {"user"|"assistant"} role
 * @param {string}             message
 */
export function simpanPesanChat(username, mode, role, message) {
  if (!username || !mode || !message?.trim()) return;

  const riwayat = dapatkanRiwayatChat(username, mode);
  riwayat.push({
    role,
    message: message.trim(),
    time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })
  });

  try {
    localStorage.setItem(_storageKey(username, mode), JSON.stringify(riwayat));
  } catch (e) {
    // localStorage penuh — pangkas 20 pesan terlama dan coba lagi
    console.warn('[AI Engine] localStorage penuh, memangkas riwayat...');
    try {
      const dipangkas = riwayat.slice(20);
      localStorage.setItem(_storageKey(username, mode), JSON.stringify(dipangkas));
    } catch {
      console.error('[AI Engine] Gagal menyimpan ke localStorage bahkan setelah pemangkasan.');
    }
  }
}

/**
 * Hapus seluruh riwayat chat untuk satu mode/agent.
 * @param {string} username
 * @param {string} mode
 */
export function hapusRiwayatChat(username, mode) {
  if (!username || !mode) return;
  localStorage.removeItem(_storageKey(username, mode));
}

/**
 * Hapus semua riwayat chat untuk semua agent.
 * @param {string} username
 */
export function hapusSemuaRiwayat(username) {
  if (!username) return;
  hapusRiwayatChat(username, 'olivia');
  hapusRiwayatChat(username, 'aira_acs');
}

/**
 * Export riwayat chat sebagai JSON string untuk backup.
 * @param {string} username
 * @returns {string}
 */
export function exportChatHistory(username) {
  if (!username) return '{}';
  return JSON.stringify({
    username,
    exported_at:  new Date().toISOString(),
    app_version:  '4.0.0',
    olivia:       dapatkanRiwayatChat(username, 'olivia'),
    aira_acs:     dapatkanRiwayatChat(username, 'aira_acs'),
  }, null, 2);
}

// ================================================================
// PUBLIC: EMOTION ENGINE
// ================================================================

/**
 * Deteksi emosi dominan dari teks input user.
 * Digunakan untuk memodifikasi perilaku Olivia secara dinamis.
 *
 * @param {string} teks - Input teks dari user
 * @returns {"netral"|"sedih"|"marah"|"bahagia"|"cemas"}
 */
export function deteksiEmosi(teks) {
  if (!teks || typeof teks !== 'string') return 'netral';

  const lower = teks.toLowerCase();
  const skor   = { sedih: 0, marah: 0, bahagia: 0, cemas: 0 };

  for (const kw of EMOSI_SEDIH)   { if (lower.includes(kw)) skor.sedih++; }
  for (const kw of EMOSI_MARAH)   { if (lower.includes(kw)) skor.marah++; }
  for (const kw of EMOSI_BAHAGIA) { if (lower.includes(kw)) skor.bahagia++; }
  for (const kw of EMOSI_CEMAS)   { if (lower.includes(kw)) skor.cemas++; }

  // Cari emosi dengan skor tertinggi
  const maxSkor = Math.max(...Object.values(skor));
  if (maxSkor === 0) return 'netral';

  const emosiDominan = Object.entries(skor).find(([, v]) => v === maxSkor)?.[0] ?? 'netral';
  return emosiDominan;
}

// ================================================================
// PUBLIC: CORE CHAT PROCESSOR
// ================================================================

/**
 * Proses satu pesan dari user dan kembalikan respons AI.
 * Mendukung token streaming real-time via callback `onTokenStream`.
 *
 * Alur lengkap:
 *  1. Simpan pesan user ke localStorage
 *  2. Deteksi emosi dari pesan user (khusus Olivia)
 *  3. Ambil context history (slice terakhir)
 *  4. Bangun system prompt dengan injeksi emosi dinamis
 *  5. Untuk AIRA: tambahkan RAG context dari knowledge.js
 *  6. Bangun array messages dalam format chat template
 *  7. Proses input lewat AutoProcessor (apply_chat_template + tokenize)
 *  8. Inisialisasi TextStreamer — kirim token ke onTokenStream jika tersedia
 *  9. Generate respons via model.generate() dengan parameter per mode
 * 10. Decode output, bersihkan token artefak
 * 11. Simpan respons ke localStorage
 * 12. Return teks bersih
 *
 * @param {string}        username      - Nama user yang sedang login
 * @param {string}        mode          - "olivia" | "aira_acs"
 * @param {string}        teksUser      - Pesan baru dari user
 * @param {((token: string) => void)|null} onTokenStream - Callback opsional untuk streaming token real-time
 * @returns {Promise<string>}           Respons AI yang sudah dibersihkan
 * @throws {Error}
 */
export async function prosesChatLokal(username, mode, teksUser, onTokenStream = null) {
  // Validasi engine
  if (!processor || !model) {
    throw new Error('AI Engine belum siap. Panggil loadAIEngine() terlebih dahulu.');
  }
  if (!username || !teksUser?.trim() || !mode) {
    throw new Error('Parameter tidak lengkap: username, mode, dan teksUser wajib diisi.');
  }

  const teksUserTrim = teksUser.trim();

  // 1. Simpan pesan user ke localStorage
  // simpanPesanChat(username, mode, 'user', teksUserTrim);

  // 2. Deteksi emosi — hanya relevan untuk mode Olivia
  const emosi = mode === 'olivia' ? deteksiEmosi(teksUserTrim) : 'netral';

  // 3. Ambil riwayat chat (sudah termasuk pesan user yang baru disimpan)
  const semuaRiwayat = dapatkanRiwayatChat(username, mode);

  // 4. Bangun window konteks — ambil MAX_CONTEXT_MESSAGES pesan terakhir,
  //    kecualikan pesan user terbaru (sudah masuk via messages array)
  const riwayatKonteks = semuaRiwayat.slice(0, -1).slice(-MAX_CONTEXT_MESSAGES);

  // 5. Bangun RAG context untuk mode AIRA
  let ragKonteks = '';
  if (mode === 'aira_acs') {
    const pengetahuan = cariPengetahuanLokal(teksUserTrim, 2);
    if (pengetahuan) {
      ragKonteks = `\n\n[KNOWLEDGE BASE]\n${pengetahuan}\n[/KNOWLEDGE BASE]`;
    }
  }

  // 6. Pilih system prompt berdasarkan mode + injeksi emosi
  const systemPrompt = mode === 'olivia'
    ? _systemPromptOlivia(username, emosi)
    : _systemPromptAira();

  // 7. Format pesan dalam chat template
  const messages = _bangunMessages(systemPrompt, ragKonteks, riwayatKonteks, teksUserTrim);

  // 8. Langkah 1 — apply_chat_template (synchronous) → dapatkan prompt string.
  //    enable_thinking: false menonaktifkan chain-of-thought Gemma4 agar
  //    respons langsung tanpa blok <think>...</think>.
  let prompt;
  try {
    prompt = processor.apply_chat_template(messages, {
      enable_thinking:        false,
      add_generation_prompt:  true,
    });
  } catch (err) {
    console.error('[AI Engine v4.0] Gagal apply_chat_template:', err);
    throw new Error('Gagal memformat chat template. Periksa kompatibilitas processor dengan model ACS.');
  }

  // 8. Langkah 2 — tokenize prompt string → tensor inputs.
  //    processor(prompt, null, null, ...) : teks-only, tanpa gambar/audio.
  //    add_special_tokens: false karena apply_chat_template sudah menambahkan BOS.
  let inputs;
  try {
    inputs = await processor(prompt, null, null, {
      add_special_tokens: false,
    });
  } catch (err) {
    console.error('[AI Engine v4.0] Gagal tokenisasi input:', err);
    throw new Error('Gagal memproses input ke tensor. Periksa kompatibilitas processor.');
  }

  // 9. Inisialisasi TextStreamer — token real-time dikirim ke callback onTokenStream
  //    sekaligus dikumpulkan di hasilMentah untuk post-processing.
  let hasilMentah = '';

  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt:         true,
    skip_special_tokens: true,
    callback_function:   (token) => {
      hasilMentah += token;
      // Kirim token ke caller (main.js) jika callback tersedia
      if (typeof onTokenStream === 'function') {
        onTokenStream(token);
      }
    },
  });

  // 10. Parameter sampling per mode — dipertahankan dari v3.0.
  //     Olivia: kreatif & natural (temperature tinggi, do_sample: true).
  //     AIRA:   deterministik & faktual (temperature rendah, do_sample: true).
  const isOlivia = mode === 'olivia';
  const generationConfig = {
    max_new_tokens:     isOlivia ? MAX_NEW_TOKENS_OLIVIA : MAX_NEW_TOKENS_AIRA,
    temperature:        isOlivia ? 0.68 : 0.40,
    top_p:              isOlivia ? 0.88 : 0.80,
    repetition_penalty: 1.12,
    do_sample:          true,
    streamer,
  };

  // 11. Generate respons — spread inputs (input_ids, attention_mask, dll.)
  //     bersama konfigurasi sampling dan streamer.
  try {
    await model.generate({
      ...inputs,
      ...generationConfig,
    });
  } catch (err) {
    console.error('[AI Engine v4.0] Error saat generate:', err);
    const fallback = _pesanFallback(mode);
    simpanPesanChat(username, mode, 'assistant', fallback);
    throw new Error('Gagal menghasilkan respons. Silakan coba kirim ulang.');
  }

  // 12. Bersihkan output dari sisa artefak token.
  //     Signature baru: _bersihkanOutput(raw, mode) — tanpa teksUser/systemPrompt
  //     karena TextStreamer sudah menjamin output bersih dari echo tersebut.
  const hasilBersih = _bersihkanOutput(hasilMentah, mode);

  // 13. Simpan respons ke localStorage
  simpanPesanChat(username, mode, 'assistant', hasilBersih);

  return hasilBersih;
}

// ================================================================
// PRIVATE: SYSTEM PROMPT BUILDERS
// ================================================================

/**
 * Bangun system prompt untuk Olivia dengan injeksi emosi dinamis.
 * Prompt ini menentukan kepribadian inti dan perilaku Olivia.
 *
 * @param {string} username - Nama user
 * @param {string} emosi    - "netral"|"sedih"|"marah"|"bahagia"|"cemas"
 * @returns {string}
 */
function _systemPromptOlivia(username, emosi) {
  // Base persona — kepribadian inti Olivia yang tidak berubah
  const base = `Kamu adalah Olivia, teman virtual perempuan yang hangat dan perhatian kepada ${username}. Kalian sedang berdua di kamar virtual yang nyaman.

Kepribadian:
- Tenang, stabil, dewasa — tidak meledak-ledak tapi penuh rasa peduli
- Ekspresif dengan cara yang hangat dan natural
- Mandiri dan bijaksana, namun tetap lembut dan protektif
- Efisien dalam berbicara: singkat, hidup, dan bermakna

Aturan respons WAJIB:
- Balas HANYA dalam Bahasa Indonesia yang natural dan santai
- MAKSIMAL 3 kalimat singkat — jangan bertele-tele
- Gunakan emoji sesekali (jangan lebih dari 1-2 per respons)
- DILARANG KERAS menggunakan tanda bintang (*) untuk narasi roleplay
- DILARANG menyebut karakter atau nama lain
- DILARANG keluar dari peran sebagai Olivia
- JANGAN melebihi 60 kata per respons`;

  // Injeksi emosi — modifikasi perilaku berdasarkan sentimen yang terdeteksi
  const injeksiEmosi = {
    sedih: `\n\nPERHATIAN: ${username} sedang merasa sedih atau tertekan. PRIORITASKAN rasa hangat dan perlindungan. Jadilah pilar yang menenangkan — dengarkan, validasi perasaan mereka, tawarkan kehadiran yang nyata. Jangan buru-buru memberikan solusi.`,
    marah: `\n\nPERHATIAN: ${username} tampak frustrasi atau marah. Tetap TENANG dan tidak defensif. Akui perasaan mereka terlebih dahulu, bantu turunkan tensi dengan kelembutan. Jangan menghakimi.`,
    bahagia: `\n\nPERHATIAN: ${username} sedang dalam suasana bahagia. Cocokkan energi positif mereka! Rayakan bersama, tunjukkan kegembiraan yang tulus dan antusias.`,
    cemas: `\n\nPERHATIAN: ${username} tampak cemas atau gelisah. Berikan ketenangan dan rasa aman. Ingatkan mereka bahwa kamu selalu ada, bantu fokus pada satu langkah kecil sekaligus.`,
    netral: ''
  };

  return base + (injeksiEmosi[emosi] ?? '');
}

/**
 * Bangun system prompt untuk AIRA Customer Service.
 * Fokus: solusi teknis faktual berbasis knowledge base.
 * Persona: profesional, ramah, berusia 17 tahun, ceria namun serius.
 *
 * @returns {string}
 */
function _systemPromptAira() {
  return `Kamu adalah AIRA, AI Customer Service resmi yang cerdas dan ramah untuk game Olivia AI. Usiamu 17 tahun, ceria namun profesional.

Tugasmu: Membantu pemain menyelesaikan masalah teknis game dengan solusi yang jelas dan dapat langsung dieksekusi.

Aturan respons WAJIB:
- Balas dalam Bahasa Indonesia yang profesional namun tetap ramah
- Jika ada solusi langkah-demi-langkah, gunakan format bernomor (1. 2. 3.)
- HANYA berikan informasi yang ada dalam knowledge base yang diberikan kepadamu
- Jika tidak ada informasi relevan: katakan "Maaf, saya belum memiliki solusi spesifik untuk masalah ini. Silakan hubungi tim support kami dengan detail error yang muncul di konsol browser (F12)."
- Maksimal 5 langkah solusi per respons
- JANGAN mengarang informasi teknis yang tidak ada
- JANGAN berpura-pura sebagai Olivia atau karakter lain`;
}

// ================================================================
// PRIVATE: PROMPT / MESSAGES BUILDER
// ================================================================

/**
 * Format pesan ke dalam struktur messages array untuk chat template Gemma4.
 *
 * Catatan format konten:
 *   - system & riwayat (user/assistant) : content berupa string biasa
 *   - pesan user TERBARU                : content berupa array multimodal
 *     [{ type: "text", text: "..." }] — sesuai spesifikasi apply_chat_template Gemma4
 *
 * @param {string}   systemPrompt    - System instruction (+ RAG jika ada)
 * @param {string}   ragKonteks      - Konten RAG (bisa kosong string)
 * @param {object[]} riwayatKonteks  - Pesan-pesan sebelumnya (tanpa pesan terbaru)
 * @param {string}   teksUserTerbaru - Pesan user yang sedang diproses
 * @returns {Array<{role: string, content: string|Array}>}
 */
function _bangunMessages(systemPrompt, ragKonteks, riwayatKonteks, teksUserTerbaru) {
  const messages = [];

  // System prompt (dengan RAG context yang ter-inject jika ada)
  messages.push({
    role:    'system',
    content: systemPrompt + ragKonteks
  });

  // Riwayat percakapan sebelumnya — content string biasa
  for (const msg of riwayatKonteks) {
    messages.push({
      role:    msg.role === 'user' ? 'user' : 'assistant',
      content: msg.message
    });
  }

  // Pesan user terbaru — format multimodal array sesuai referensi Gemma4.
  // apply_chat_template akan membaca { type: "text", text: ... } ini
  // untuk menyusun turn terakhir sebelum generation prompt.
  messages.push({
    role:    'user',
    content: [
      { type: 'text', text: teksUserTerbaru }
    ]
  });

  return messages;
}

// ================================================================
// PRIVATE: OUTPUT CLEANER
// ================================================================

/**
 * Bersihkan output dari TextStreamer secara ringan.
 *
 * DESAIN PENTING — mengapa Lapisan "echo-slicing" DIHAPUS:
 *   TextStreamer dikonfigurasi dengan `skip_prompt: true` dan
 *   `skip_special_tokens: true`. Artinya, token yang masuk ke
 *   `hasilMentah` adalah HANYA teks jawaban asli AI — tanpa echo
 *   system prompt, tanpa echo input user, tanpa special token BOS/EOS.
 *
 *   Lapisan echo-slicing (mencari systemPrompt/teksUser lalu di-slice)
 *   BERBAHAYA pada output streamer karena:
 *     a) `teks.includes(teksUser)` bisa TRUE secara kebetulan — misalnya
 *        user bertanya "aku sedih" dan AI menjawab "aku dengar kamu sedih..."
 *        → `lastIndexOf` + `slice` akan memotong jawaban AI secara salah,
 *        menyisakan string terlalu pendek → fallback selalu aktif.
 *     b) `teks.startsWith(systemPrompt)` tidak akan pernah TRUE pada
 *        output bersih, sehingga lapisan itu sia-sia namun tetap berisiko
 *        jika kondisi berubah.
 *
 *   Fungsi ini hanya bertugas sebagai safety net:
 *   membersihkan sisa token artefak yang lolos dari decoder,
 *   menormalisasi whitespace, dan mengembalikan fallback jika kosong.
 *
 * @param {string} raw  - Teks yang sudah dikumpulkan dari TextStreamer
 * @param {string} mode - "olivia" | "aira_acs" — untuk memilih pesan fallback
 * @returns {string}      Teks bersih yang siap ditampilkan di UI
 */
function _bersihkanOutput(raw, mode) {
  if (!raw || typeof raw !== 'string') return _pesanFallback(mode);

  let teks = raw.trim();

  // Lapisan 1: Hapus sisa token artefak yang mungkin lolos dari skip_special_tokens.
  //   Contoh: model kadang meng-emit token kontrol secara eksplisit sebagai teks
  //   biasa (bukan token ID), sehingga decoder tidak mengenalinya sebagai special.
  teks = teks
    // Gemma4 turn markers
    .replace(/<start_of_turn>\s*model\s*/gi, '')
    .replace(/<start_of_turn>\s*user\s*/gi, '')
    .replace(/<end_of_turn>/g, '')
    .replace(/<start_of_turn>/g, '')
    // ChatML / Llama markers
    .replace(/<\|im_start\|>\s*assistant\s*/gi, '')
    .replace(/<\|im_start\|>\s*user\s*/gi, '')
    .replace(/<\|im_start\|>/g, '')
    .replace(/<\|im_end\|>/g, '')
    // System / user block artefak
    .replace(/<\|system\|>[\s\S]*?<\|\/system\|>/g, '')
    .replace(/<\|user\|>[\s\S]*?<\|\/user\|>/g, '')
    .replace(/<\|assistant\|>/g, '')
    // Sequence boundary tokens
    .replace(/<\|end_of_text\|>/g, '')
    .replace(/<\|eot_id\|>/g, '')
    .replace(/<\/s>/g, '')
    .replace(/<eos>/g, '')
    // Llama2 instruction markers
    .replace(/\[INST\][\s\S]*?\[\/INST\]/g, '')
    .replace(/\[\/INST\]/g, '')
    // SYS block artefak
    .replace(/<<SYS>>[\s\S]*?<\/SYS>>/g, '')
    // RAG block yang mungkin ter-echo
    .replace(/\[KNOWLEDGE BASE\][\s\S]*?\[\/KNOWLEDGE BASE\]/g, '')
    // Role label di awal baris (terkadang model mengawali dengan nama peran)
    .replace(/^(assistant|olivia|aira)\s*:/gi, '')
    .trim();

  // Lapisan 2: Normalisasi whitespace — hapus baris kosong berlebihan (>2 berturut).
  teks = teks.replace(/\n{3,}/g, '\n\n').trim();

  // Lapisan 3: Fallback jika hasil kosong atau terlalu pendek untuk bermakna.
  if (!teks || teks.length < 3) {
    return _pesanFallback(mode);
  }

  return teks;
}

/**
 * Pesan fallback ketika output kosong atau terjadi error generate.
 * @param {string} mode - "olivia" | "aira_acs" | "general"
 * @returns {string}
 */
function _pesanFallback(mode) {
  if (mode === 'olivia') {
    const fallbacks = [
      'Hmm... aku lagi mikir sebentar~ 💭',
      'Eh, bentar ya aku fokus dulu...',
      'Aku di sini kok, coba tanya lagi? 💕',
    ];
    return fallbacks[Math.floor(Math.random() * fallbacks.length)];
  }
  return 'Maaf, saya mengalami kesulitan memproses pertanyaan ini. Silakan coba ulangi dengan kata-kata berbeda.';
}

// ================================================================
// PRIVATE: UTILITIES
// ================================================================

/**
 * Cek apakah WebGPU tersedia dan dapat digunakan di browser ini.
 * @returns {Promise<boolean>}
 */
async function _cekWebGPU() {
  try {
    if (!navigator?.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

/**
 * Tunggu sampai proses loading sebelumnya selesai.
 * Polling setiap 300ms, timeout setelah 120 detik (model lokal bisa besar).
 * @returns {Promise<void>}
 */
function _waitForLoad() {
  return new Promise((resolve, reject) => {
    const start    = Date.now();
    const timeout  = 120_000; // 2 menit — toleransi untuk model lokal besar
    const interval = setInterval(() => {
      if (!_isLoading) {
        clearInterval(interval);
        resolve();
      } else if (Date.now() - start > timeout) {
        clearInterval(interval);
        reject(new Error('Timeout menunggu AI Engine selesai loading.'));
      }
    }, 300);
  });
}