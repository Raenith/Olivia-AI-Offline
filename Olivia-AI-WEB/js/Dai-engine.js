/**
 * ================================================================
 * OLIVIA AI — AI Engine v3.0 (ai-engine.js)
 *
 * Features:
 *   - Transformers.js v3 (Client-Side Edge AI)
 *   - WebGPU acceleration + WASM fallback
 *   - Multi-Agent: Olivia (companion) + AIRA ACS (customer service)
 *   - Emotion Engine: deteksi sentimen → injeksi perilaku dinamis
 *   - Local RAG: keyword-matching dari knowledge.js
 *   - Multi-turn memory via localStorage (keyed by username+mode)
 *   - Token window management: slice(-MAX_WINDOW)
 *
 * Model: onnx-community/gemma-4-E2B-it-ONNX (GGUF/ONNX q4)
 * Storage: localStorage browser (100% offline, privat)
 * Offline: Ya — setelah download model pertama kali
 * ================================================================
 */

import { cariPengetahuanLokal } from './knowledge.js';

// ================================================================
// CONSTANTS
// ================================================================

/**
 * ID model dari HuggingFace Hub (ONNX quantized).
 * Alternatif: 'onnx-community/gemma-2-2b-it-ONNX' untuk kualitas lebih tinggi
 * tapi membutuhkan VRAM lebih besar.
 */
const MODEL_ID = 'onnx-community/Phi-3.5-mini-instruct-onnx-web';

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
 * Lebih dari ini akan menyebabkan token overflow pada model 1B.
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

/** Instance pipeline Transformers.js — null sebelum diinisialisasi */
let generator = null;

/** Flag loading untuk mencegah double-init */
let _isLoading = false;

// ================================================================
// PUBLIC: AI INITIALIZATION
// ================================================================

/**
 * Inisialisasi AI Engine (singleton).
 * Memuat model ONNX ke WebGPU/WASM — hanya terjadi sekali per sesi.
 *
 * @param {(status: string) => void} onProgress - Callback update teks status ke UI
 * @returns {Promise<void>}
 * @throws {Error} Jika browser tidak kompatibel atau model gagal dimuat
 */
export async function loadAIEngine(onProgress = () => {}) {
  // Jika sudah siap, langsung return
  if (generator !== null) {
    onProgress('🟢 AI Engine aktif — 100% Offline');
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
    onProgress('📦 Mengimpor Transformers.js v3...');

    // Dinamis import Transformers.js v3 dari CDN (lazy-loading)
    const { pipeline, env } = await import(
      'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0'
    );

    // Konfigurasi environment
    // Proxy WASM diaktifkan untuk mencegah blocking thread utama
    env.backends.onnx.wasm.proxy       = true;
    env.backends.onnx.wasm.numThreads  = navigator.hardwareConcurrency
      ? Math.min(navigator.hardwareConcurrency, 4)
      : 2;

    // Cache model di IndexedDB browser (lebih persistent dari WASM)
    env.cacheDir = './.cache';

    onProgress(`⚙️ Memuat model ${MODEL_ID}...`);
    onProgress('💡 Download hanya diperlukan sekali (~1.2 GB). Selanjutnya 100% offline.');

    // Tentukan device: WebGPU jika tersedia, fallback ke WASM CPU
    const device = await _cekWebGPU() ? 'webgpu' : 'wasm';
    onProgress(device === 'webgpu'
      ? '🖥️ WebGPU terdeteksi — menggunakan akselerasi GPU...'
      : '💻 WebGPU tidak tersedia — menggunakan mode CPU (WASM)...'
    );

    // Inisialisasi pipeline text-generation
    generator = await pipeline('text-generation', MODEL_ID, {
      dtype:    'q4f16',   // Quantisasi 4-bit float16 — optimal untuk 1B params
      device:   device,
      progress_callback: (info) => {
        if (info.status === 'downloading') {
          const pct = info.total
            ? `${Math.round((info.loaded / info.total) * 100)}%`
            : '...';
          const mb  = info.total
            ? `${(info.total / 1024 / 1024).toFixed(0)} MB`
            : '';
          onProgress(`📥 Mengunduh ${info.name ?? 'model'} ${pct} ${mb}`);
        } else if (info.status === 'initiate') {
          onProgress(`🔧 Mempersiapkan shard: ${info.name ?? 'model'}...`);
        } else if (info.status === 'done') {
          onProgress(`✅ Shard selesai: ${info.name ?? ''}`);
        } else if (info.status === 'loading') {
          onProgress('🔩 Memuat model ke memori GPU/CPU...');
        } else if (info.status === 'ready') {
          onProgress('🟢 Model siap!');
        }
      }
    });

    onProgress('🟢 AI Engine aktif — Siap 100% Offline!');

  } catch (err) {
    _isLoading = false;
    generator  = null;
    console.error('[AI Engine] Gagal inisialisasi:', err);

    // Pesan error yang ramah pengguna
    const msg = err?.message ?? String(err);
    if (msg.includes('WebGPU')) {
      throw new Error('WebGPU tidak tersedia. Gunakan Chrome/Edge terbaru dan aktifkan Hardware Acceleration.');
    } else if (msg.includes('fetch') || msg.includes('network')) {
      throw new Error('Gagal mengunduh model. Periksa koneksi internet untuk download pertama kali.');
    } else if (msg.includes('memory') || msg.includes('OOM')) {
      throw new Error('Memori tidak cukup. Tutup semua aplikasi lain dan coba lagi.');
    } else {
      throw new Error(`Gagal memuat AI Engine: ${msg}`);
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
  return generator !== null;
}

// ================================================================
// PUBLIC: CHAT HISTORY (localStorage)
// ================================================================

/**z
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
 *
 * @param {string}           username
 * @param {string}           mode
 * @param {"user"|"assistant"} role
 * @param {string}           message
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
      console.error('[AI Engine] Gagal menyimpan ke localStorage.');
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
    app_version:  '3.0.0',
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

  for (const kw of EMOSI_SEDIH)  { if (lower.includes(kw)) skor.sedih++; }
  for (const kw of EMOSI_MARAH)  { if (lower.includes(kw)) skor.marah++; }
  for (const kw of EMOSI_BAHAGIA){ if (lower.includes(kw)) skor.bahagia++; }
  for (const kw of EMOSI_CEMAS)  { if (lower.includes(kw)) skor.cemas++; }

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
 *
 * Alur lengkap:
 *  1. Simpan pesan user ke localStorage
 *  2. Deteksi emosi dari pesan user
 *  3. Ambil context history (slice terakhir)
 *  4. Bangun system prompt dengan injeksi emosi dinamis
 *  5. Untuk AIRA: tambahkan RAG context dari knowledge.js
 *  6. Format prompt akhir dalam format Gemma4-E2B-it Chat
 *  7. Generate respons via model pipeline
 *  8. Bersihkan output dari token artifact
 *  9. Simpan respons ke localStorage
 * 10. Return teks bersih
 *
 * @param {string} username  - Nama user yang sedang login
 * @param {string} mode      - "olivia" | "aira_acs"
 * @param {string} teksUser  - Pesan baru dari user
 * @returns {Promise<string>} Respons AI yang sudah dibersihkan
 * @throws {Error}
 */
export async function prosesChatLokal(username, mode, teksUser) {
  // Validasi
  if (!generator) {
    throw new Error('AI Engine belum siap. Panggil loadAIEngine() terlebih dahulu.');
  }
  if (!username || !teksUser?.trim() || !mode) {
    throw new Error('Parameter tidak lengkap.');
  }

  const teksUserTrim = teksUser.trim();

  // 1. Simpan pesan user ke localStorage
  simpanPesanChat(username, mode, 'user', teksUserTrim);

  // 2. Deteksi emosi untuk Olivia mode
  const emosi = mode === 'olivia' ? deteksiEmosi(teksUserTrim) : 'netral';

  // 3. Ambil riwayat chat (sudah termasuk pesan user yang baru)
  const semuaRiwayat = dapatkanRiwayatChat(username, mode);

  // 4. Bangun window konteks — ambil MAX_CONTEXT_MESSAGES pesan terakhir
  //    Kecualikan pesan user terbaru (sudah di dalam prompt [INST])
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

  // 7. Format pesan dalam format Gemma4-E2B-it chat template
  const messages = _bangunMessages(systemPrompt, ragKonteks, riwayatKonteks, teksUserTrim);

  // 8. Generate respons
  let hasilMentah = '';
  try {
    const output = await generator(messages, {
      max_new_tokens:     mode === 'olivia' ? MAX_NEW_TOKENS_OLIVIA : MAX_NEW_TOKENS_AIRA,
      temperature:        mode === 'olivia' ? 0.68 : 0.40,
      top_p:              mode === 'olivia' ? 0.88 : 0.80,
      repetition_penalty: 1.12,
      do_sample:          true,
      stop_sequences:     mode === 'olivia' ? STOP_SEQUENCES_OLIVIA : STOP_SEQUENCES_AIRA,
    });

    // Transformers.js v3 mengembalikan array of objects
    if (Array.isArray(output) && output.length > 0) {
      // Format v3: output[0].generated_text bisa berupa string atau array of messages
      const genText = output[0]?.generated_text;
      if (typeof genText === 'string') {
        hasilMentah = genText;
      } else if (Array.isArray(genText)) {
        // Messages format — ambil pesan assistant terakhir
        const lastMsg = [...genText].reverse().find(m => m.role === 'assistant');
        hasilMentah = lastMsg?.content ?? '';
      }
    }

  } catch (err) {
    console.error('[AI Engine] Error saat generate:', err);
    simpanPesanChat(username, mode, 'assistant', _pesanFallback(mode));
    throw new Error('Gagal menghasilkan respons. Silakan coba kirim ulang.');
  }

  // 9. Bersihkan output dari artifacts dan simpan
  const hasilBersih = _bersihkanOutput(hasilMentah, teksUserTrim, systemPrompt);

  // 10. Simpan respons ke localStorage
  simpanPesanChat(username, mode, 'assistant', hasilBersih);

  return hasilBersih;
}

// ================================================================
// PRIVATE: SYSTEM PROMPT BUILDERS
// ================================================================

/**
 * Bangun system prompt untuk Olivia dengan injeksi emosi dinamis.
 *
 * @param {string} username - Nama user
 * @param {string} emosi    - "netral"|"sedih"|"marah"|"bahagia"|"cemas"
 * @returns {string}
 */
function _systemPromptOlivia(username, emosi) {
  // Base persona
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

  // Injeksi emosi — modifikasi perilaku berdasarkan sentimen user
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
// PRIVATE: PROMPT BUILDER
// ================================================================

/**
 * Format pesan ke dalam struktur messages array untuk Transformers.js v3.
 * Format: [{ role, content }, ...]
 *
 * @param {string}  systemPrompt   - System instruction
 * @param {string}  ragKonteks     - RAG context (bisa kosong)
 * @param {object[]} riwayatKonteks - Pesan-pesan sebelumnya (kecuali yang terbaru)
 * @param {string}  teksUserTerbaru - Pesan user yang sedang diproses
 * @returns {Array<{role: string, content: string}>}
 */
function _bangunMessages(systemPrompt, ragKonteks, riwayatKonteks, teksUserTerbaru) {
  const messages = [];

  // System prompt (dengan RAG jika ada)
  messages.push({
    role:    'system',
    content: systemPrompt + ragKonteks
  });

  // Riwayat percakapan sebelumnya (context window)
  for (const msg of riwayatKonteks) {
    messages.push({
      role:    msg.role === 'user' ? 'user' : 'assistant',
      content: msg.message
    });
  }

  // Pesan user terbaru
  messages.push({
    role:    'user',
    content: teksUserTerbaru
  });

  return messages;
}

// ================================================================
// PRIVATE: OUTPUT CLEANER
// ================================================================

/**
 * Bersihkan output mentah dari model.
 * Hapus: echo prompt, special tokens, dan whitespace berlebih.
 *
 * @param {string} raw          - Output mentah dari generator
 * @param {string} teksUser     - Pesan user (untuk menghapus echo)
 * @param {string} systemPrompt - System prompt (untuk menghapus echo)
 * @returns {string}            - Teks bersih yang siap ditampilkan
 */
function _bersihkanOutput(raw, teksUser, systemPrompt) {
  if (!raw || typeof raw !== 'string') return _pesanFallback('olivia');

  let teks = raw.trim();

  // Hapus echo system prompt jika ada
  if (teks.startsWith(systemPrompt)) {
    teks = teks.slice(systemPrompt.length).trim();
  }

  // Hapus echo input user jika ada
  if (teks.includes(teksUser)) {
    const idx = teks.lastIndexOf(teksUser);
    teks = teks.slice(idx + teksUser.length).trim();
  }

  // Hapus special/control tokens
  teks = teks
    .replace(/<\|im_start\|>[\s\S]*?<\|im_end\|>/g, '')
    .replace(/<\|system\|>[\s\S]*?<\|\/system\|>/g, '')
    .replace(/<\|user\|>[\s\S]*?<\|\/user\|>/g, '')
    .replace(/<\|assistant\|>/g, '')
    .replace(/<\|end_of_text\|>/g, '')
    .replace(/<\|eot_id\|>/g, '')
    .replace(/<\/s>/g, '')
    .replace(/\[\/INST\]/g, '')
    .replace(/\[INST\][\s\S]*?\[\/INST\]/g, '')
    .replace(/<<SYS>>[\s\S]*?<\/SYS>>/g, '')
    .replace(/\[KNOWLEDGE BASE\][\s\S]*?\[\/KNOWLEDGE BASE\]/g, '')
    .replace(/assistant:/gi, '')
    .replace(/olivia:/gi, '')
    .replace(/aira:/gi, '')
    .trim();

  // Hapus baris kosong berlebihan (lebih dari 2 baris kosong berturut-turut)
  teks = teks.replace(/\n{3,}/g, '\n\n').trim();

  // Fallback jika hasil kosong atau terlalu pendek
  if (!teks || teks.length < 3) {
    return _pesanFallback('general');
  }

  return teks;
}

/**
 * Pesan fallback ketika output kosong atau terjadi error.
 * @param {string} mode
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
 * Cek apakah WebGPU tersedia di browser ini.
 * @returns {Promise<boolean>}
 */
async function _cekWebGPU() {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

/**
 * Tunggu sampai proses loading sebelumnya selesai.
 * Polling setiap 300ms, timeout setelah 60 detik.
 * @returns {Promise<void>}
 */
function _waitForLoad() {
  return new Promise((resolve, reject) => {
    const start   = Date.now();
    const timeout = 60_000;
    const interval = setInterval(() => {
      if (!_isLoading) {
        clearInterval(interval);
        resolve();
      } else if (Date.now() - start > timeout) {
        clearInterval(interval);
        reject(new Error('Timeout menunggu AI Engine.'));
      }
    }, 300);
  });
}
