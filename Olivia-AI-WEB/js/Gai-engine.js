/**
 * ================================================================
 * OLIVIA AI — AI Engine v4.0 (ai-engine.js)
 *
 * Features:
 * - Transformers.js v3 (Client-Side Edge AI via esm.sh)
 * - WebGPU acceleration (Strict Local Ingestion)
 * - Multi-Agent: Olivia (companion) + AIRA ACS (customer service)
 * - Emotion Engine: deteksi sentimen → injeksi perilaku dinamis
 * - Local RAG: keyword-matching dari knowledge.js
 * - Multi-turn memory via localStorage (keyed by username+mode)
 * - Token window management: slice(-MAX_WINDOW)
 * - Real-time Token Streaming via TextStreamer
 *
 * Model: Gemma4ForConditionalGeneration (Local Quantized q4)
 * Storage: localStorage browser (100% offline, privat)
 * Offline: 100% Berjalan secara lokal melalui sistem file terarah
 * ================================================================
 */

import { cariPengetahuanLokal } from './knowledge.js';
import { env, AutoProcessor, Gemma4ForConditionalGeneration, TextStreamer } from "https://esm.sh/@huggingface/transformers";

// ================================================================
// CONSTANTS & ENVIRONMENT CONFIGURATION
// ================================================================

/**
 * ID model lokal yang ditargetkan di dalam sistem berkas lokal.
 */
const MODEL_ID = "ACS";

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

/** Instance terpisah untuk Processor dan Model v4.0 */
let processor = null;
let model = null;

/** Flag loading untuk mencegah double-init */
let _isLoading = false;

// ================================================================
// PUBLIC: AI INITIALIZATION
// ================================================================

/**
 * Inisialisasi AI Engine (singleton).
 * Memuat berkas konfigurasi processor dan model biner secara terpisah.
 *
 * @param {(status: string) => void} onProgress - Callback update teks status ke sistem utama
 * @returns {Promise<void>}
 */
export async function loadAIEngine(onProgress = () => {}) {
  // Jika arsitektur engine sudah terbentuk, langsung return
  if (processor !== null && model !== null) {
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
    onProgress('📦 Mengonfigurasi Jalur Prosedural Lokal...');

    // Konfigurasi internal environment secara ketat
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = '/gatau/tester/models/';

    onProgress(`⚙️ Memuat Model Arsitektur: ${MODEL_ID}...`);

    // Memuat modul tokenizer/processor terpisah
    processor = await AutoProcessor.from_pretrained(MODEL_ID);

    // Memuat arsitektur model komputasi terpisah dengan alokasi WebGPU eksklusif
    model = await Gemma4ForConditionalGeneration.from_pretrained(MODEL_ID, {
      dtype: "q4",
      device: "webgpu",
      progress_callback: (info) => {
        if (info.status === 'downloading') {
          const pct = info.total
            ? `${Math.round((info.loaded / info.total) * 100)}%`
            : '...';
          const mb  = info.total
            ? `${(info.total / 1024 / 1024).toFixed(0)} MB`
            : '';
          onProgress(`📥 Memuat dependensi runtime ${info.name ?? 'model'} ${pct} ${mb}`);
        } else if (info.status === 'initiate') {
          onProgress(`🔧 Mempersiapkan alokasi shard: ${info.name ?? 'model'}...`);
        } else if (info.status === 'done') {
          onProgress(`✅ Shard berhasil dialokasikan: ${info.name ?? ''}`);
        } else if (info.status === 'loading') {
          onProgress('🔩 Mengompilasi struktur kernel ke VRAM WebGPU...');
        } else if (info.status === 'ready') {
          onProgress('🟢 Sinkronisasi model selesai.');
        }
      }
    });

    onProgress('🟢 AI Engine aktif — Siap 100% Offline!');

  } catch (err) {
    _isLoading = false;
    processor = null;
    model = null;
    console.error('[AI Engine] Gagal inisialisasi:', err);

    const msg = err?.message ?? String(err);
    if (msg.includes('WebGPU')) {
      throw new Error('WebGPU tidak dapat diinisialisasi. Pastikan subsistem akselerasi hardware aktif.');
    } else {
      throw new Error(`Gagal memuat AI Engine: ${msg}`);
    }
  } finally {
    _isLoading = false;
  }
}

/**
 * Cek apakah subsistem inti AI Engine telah siap digunakan.
 * @returns {boolean}
 */
export function isAIReady() {
  return processor !== null && model !== null;
}

// ================================================================
// PUBLIC: CHAT HISTORY (localStorage Management)
// ================================================================

function _storageKey(username, mode) {
  const safeName = String(username).replace(/[^a-zA-Z0-9_\-ก-๙ก-ฮ]/g, '_').slice(0, 32);
  const safeMode = String(mode).replace(/[^a-zA-Z0-9_]/g, '_');
  return `olivia_chat_${safeName}_${safeMode}`;
}

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
    console.warn('[AI Engine] localStorage penuh, memangkas riwayat...');
    try {
      const dipangkas = riwayat.slice(20);
      localStorage.setItem(_storageKey(username, mode), JSON.stringify(dipangkas));
    } catch {
      console.error('[AI Engine] Gagal menyimpan ke localStorage.');
    }
  }
}

export function hapusRiwayatChat(username, mode) {
  if (!username || !mode) return;
  localStorage.removeItem(_storageKey(username, mode));
}

export function hapusSemuaRiwayat(username) {
  if (!username) return;
  hapusRiwayatChat(username, 'olivia');
  hapusRiwayatChat(username, 'aira_acs');
}

export function exportChatHistory(username) {
  if (!username) return '{}';
  return JSON.stringify({
    username,
    exported_at:  new Date().toISOString(),
    app_version:  '4.0.0',
    olivia:        dapatkanRiwayatChat(username, 'olivia'),
    aira_acs:      dapatkanRiwayatChat(username, 'aira_acs'),
  }, null, 2);
}

// ================================================================
// PUBLIC: EMOTION ENGINE
// ================================================================

export function deteksiEmosi(teks) {
  if (!teks || typeof teks !== 'string') return 'netral';

  const lower = teks.toLowerCase();
  const skor   = { sedih: 0, marah: 0, bahagia: 0, cemas: 0 };

  for (const kw of EMOSI_SEDIH)  { if (lower.includes(kw)) skor.sedih++; }
  for (const kw of EMOSI_MARAH)  { if (lower.includes(kw)) skor.marah++; }
  for (const kw of EMOSI_BAHAGIA){ if (lower.includes(kw)) skor.bahagia++; }
  for (const kw of EMOSI_CEMAS)  { if (lower.includes(kw)) skor.cemas++; }

  const maxSkor = Math.max(...Object.values(skor));
  if (maxSkor === 0) return 'netral';

  const emosiDominan = Object.entries(skor).find(([, v]) => v === maxSkor)?.[0] ?? 'netral';
  return emosiDominan;
}

// ================================================================
// PUBLIC: CORE CHAT PROCESSOR (With Stream Pipeline)
// ================================================================

/**
 * Memproses pesan, menerapkan templat instruksi, serta melemparkan output token
 * secara real-time apabila fungsi callback onTokenStream disuplai.
 *
 * @param {string} username
 * @param {string} mode
 * @param {string} teksUser
 * @param {Function|null} onTokenStream - Callback penangkap pecahan token teks real-time
 * @returns {Promise<string>}
 */
export async function prosesChatLokal(username, mode, teksUser, onTokenStream = null) {
  if (!processor || !model) {
    throw new Error('AI Engine belum siap. Panggil loadAIEngine() terlebih dahulu.');
  }
  if (!username || !teksUser?.trim() || !mode) {
    throw new Error('Parameter tidak lengkap.');
  }

  const teksUserTrim = teksUser.trim();

  // 1. Simpan pesan user ke localStorage
  simpanPesanChat(username, mode, 'user', teksUserTrim);

  // 2. Deteksi komponen emosi
  const emosi = mode === 'olivia' ? deteksiEmosi(teksUserTrim) : 'netral';

  // 3. Ambil riwayat chat teranyar
  const semuaRiwayat = dapatkanRiwayatChat(username, mode);

  // 4. Batasi riwayat konteks berdasarkan batasan window maksimum
  const riwayatKonteks = semuaRiwayat.slice(0, -1).slice(-MAX_CONTEXT_MESSAGES);

  // 5. Injeksi modul RAG jika agen merupakan AIRA
  let ragKonteks = '';
  if (mode === 'aira_acs') {
    const pengetahuan = cariPengetahuanLokal(teksUserTrim, 2);
    if (pengetahuan) {
      ragKonteks = `\n\n[KNOWLEDGE BASE]\n${pengetahuan}\n[/KNOWLEDGE BASE]`;
    }
  }

  // 6. Siapkan System Prompt terstruktur
  const systemPrompt = mode === 'olivia'
    ? _systemPromptOlivia(username, emosi)
    : _systemPromptAira();

  // 7. Bangun format array pesan formal
  const messages = _bangunMessages(systemPrompt, ragKonteks, riwayatKonteks, teksUserTrim);

  // 8. Eksekusi inferensi model
  let hasilMentah = '';
  try {
    // Menerapkan chat template terstruktur bawaan model melalui processor
    const formattedPrompt = processor.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
    const inputs = await processor(formattedPrompt);

    // Inisialisasi mekanisme streaming token jika callback aktif tersedia
    let streamer = null;
    if (typeof onTokenStream === 'function') {
      streamer = new TextStreamer(processor, {
        skip_prompt: true,
        callback_function: (tokenText) => {
          onTokenStream(tokenText);
        }
      });
    }

    // Bangun parameter generate yang kompatibel penuh dengan Transformers.js v3
    const generateParams = {
      ...inputs,
      max_new_tokens:     mode === 'olivia' ? MAX_NEW_TOKENS_OLIVIA : MAX_NEW_TOKENS_AIRA,
      temperature:        mode === 'olivia' ? 0.68 : 0.40,
      top_p:              mode === 'olivia' ? 0.88 : 0.80,
      repetition_penalty: 1.12,
      do_sample:          true,
      stop_strings:       mode === 'olivia' ? STOP_SEQUENCES_OLIVIA : STOP_SEQUENCES_AIRA,
    };

    if (streamer) {
      generateParams.streamer = streamer;
    }

    const outputIds = await model.generate(generateParams);
    
    // Dekode seluruh ID token menjadi string mentah utuh (diperlukan untuk pembersihan echo prompt)
    hasilMentah = processor.batch_decode(outputIds, { skip_special_tokens: false })[0];

  } catch (err) {
    console.error('[AI Engine] Error saat generate:', err);
    simpanPesanChat(username, mode, 'assistant', _pesanFallback(mode));
    throw new Error('Gagal menghasilkan respons. Silakan coba kirim ulang.');
  }

  // 9. Eksekusi pembersihan berlapis
  const hasilBersih = _bersihkanOutput(hasilMentah, teksUserTrim, systemPrompt);

  // 10. Amankan hasil bersih ke subsistem penyimpanan lokal
  simpanPesanChat(username, mode, 'assistant', hasilBersih);

  return hasilBersih;
}

// ================================================================
// PRIVATE: SYSTEM PROMPT BUILDERS
// ================================================================

function _systemPromptOlivia(username, emosi) {
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

  const injeksiEmosi = {
    sedih: `\n\nPERHATIAN: ${username} sedang merasa sedih atau tertekan. PRIORITASKAN rasa hangat dan perlindungan. Jadilah pilar yang menenangkan — dengarkan, validasi perasaan mereka, tawarkan kehadiran yang nyata. Jangan buru-buru memberikan solusi.`,
    marah: `\n\nPERHATIAN: ${username} tampak frustrasi atau marah. Tetap TENANG dan tidak defensif. Akui perasaan mereka terlebih dahulu, bantu turunkan tensi dengan kelembutan. Jangan menghakimi.`,
    bahagia: `\n\nPERHATIAN: ${username} sedang dalam suasana bahagia. Cocokkan energi positif mereka! Rayakan bersama, tunjukkan kegembiraan yang tulus dan antusias.`,
    cemas: `\n\nPERHATIAN: ${username} tampak cemas atau gelisah. Berikan ketenangan dan rasa aman. Ingatkan mereka bahwa kamu selalu ada, bantu fokus pada satu langkah kecil sekaligus.`,
    netral: ''
  };

  return base + (injeksiEmosi[emosi] ?? '');
}

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

function _bangunMessages(systemPrompt, ragKonteks, riwayatKonteks, teksUserTerbaru) {
  const messages = [];

  messages.push({
    role:    'system',
    content: systemPrompt + ragKonteks
  });

  for (const msg of riwayatKonteks) {
    messages.push({
      role:    msg.role === 'user' ? 'user' : 'assistant',
      content: msg.message
    });
  }

  messages.push({
    role:    'user',
    content: teksUserTerbaru
  });

  return messages;
}

// ================================================================
// PRIVATE: OUTPUT CLEANER
// ================================================================

function _bersihkanOutput(raw, teksUser, systemPrompt) {
  if (!raw || typeof raw !== 'string') return _pesanFallback('olivia');

  let teks = raw.trim();

  // Memotong deretan teks akibat echo sistem prompt bawaan generasi mentah
  if (teks.startsWith(systemPrompt)) {
    teks = teks.slice(systemPrompt.length).trim();
  }

  // Memotong input user yang memantul kembali pada stream sequence
  if (teks.includes(teksUser)) {
    const idx = teks.lastIndexOf(teksUser);
    teks = teks.slice(idx + teksUser.length).trim();
  }

  // Filtrasi komprehensif terhadap residu token kontrol arsitektur model
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

  // Restrukturisasi jeda baris baru yang berlebihan
  teks = teks.replace(/\n{3,}/g, '\n\n').trim();

  if (!teks || teks.length < 3) {
    return _pesanFallback('general');
  }

  return teks;
}

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
        reject(new Error('Timeout menunggu inisialisasi subsistem engine lokal.'));
      }
    }, 300);
  });
}