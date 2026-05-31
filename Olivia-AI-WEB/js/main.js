/**
 * ================================================================
 * OLIVIA AI — Main UI Controller v3.0 (main.js)
 *
 * Tanggung jawab:
 *   - Login / Logout (localStorage persisted)
 *   - Mode switching: Olivia ↔ AIRA ACS
 *   - Chat render & auto-scroll
 *   - Loading overlay management
 *   - Thinking bubble animation
 *   - Emotion-driven character visual feedback
 *   - Status bar updates
 *   - Keyboard shortcuts
 *   - Error display in-chat
 *
 * Depends on: ai-engine.js
 * ================================================================
 */

import {
  loadAIEngine,
  isAIReady,
  prosesChatLokal,
  dapatkanRiwayatChat,
  simpanPesanChat,
  hapusRiwayatChat,
  deteksiEmosi,
} from './ai-engine.js';

// ================================================================
// APPLICATION STATE
// ================================================================

const state = {
  /** Username yang sedang login (null = belum login) */
  username:     null,

  /** Mode agent aktif */
  mode:         'olivia', // "olivia" | "aira_acs"

  /** Apakah model AI sudah berhasil dimuat */
  aiLoaded:     false,

  /** Apakah sedang menunggu respons AI (mutex lock UI) */
  isProcessing: false,

  /** Emosi terakhir yang terdeteksi */
  emosiTerakhir: 'netral',
};

// ================================================================
// DOM HELPERS
// ================================================================

/** Ambil elemen DOM secara aman */
const el = (id) => document.getElementById(id);

/** Tambah class 'hidden' */
const hide = (id) => el(id)?.classList.add('hidden');

/** Hapus class 'hidden' */
const show = (id) => el(id)?.classList.remove('hidden');

/** Cek apakah elemen tersembunyi */
const isHidden = (id) => el(id)?.classList.contains('hidden') ?? true;

// ================================================================
// ENTRY POINT
// ================================================================

document.addEventListener('DOMContentLoaded', () => {
  _initApp();
});

function _initApp() {
  _periksakLoginTersimpan();
  _pasangSemuaEventListener();
  _pasangKeyboardShortcuts();

  _startBlinkSystem();
  _startThinkingAnimation();
}

// ================================================================
// LOGIN / LOGOUT
// ================================================================

/**
 * Cek apakah ada sesi login yang tersimpan di localStorage.
 * Jika ada, langsung masuk ke game section.
 */
function _periksakLoginTersimpan() {
  const savedUser = localStorage.getItem('olivia_user_login');
  if (savedUser) {
    state.username = savedUser;
    _tampilkanGame();
  } else {
    _tampilkanLogin();
  }
}

/** Tampilkan layar login, sembunyikan game */
function _tampilkanLogin() {
  show('login-section');
  hide('game-section');

  // Bersihkan error sebelumnya
  const errEl = el('login-error');
  if (errEl) errEl.textContent = '';

  // Focus ke input setelah animasi muncul
  setTimeout(() => el('username-input')?.focus(), 120);
}

/** Tampilkan layar game, sembunyikan login */
function _tampilkanGame() {
  hide('login-section');
  show('game-section');

  _updateHUD();
  _renderChat();
  _setStatus('idle');

  // Pasang event listener resize untuk auto-scroll
  window.addEventListener('resize', () => _scrollChatKeBawah());

  // Focus ke input chat setelah transisi
  setTimeout(() => el('user-input')?.focus(), 150);
}

/** Handle klik tombol login */
function _handleLogin() {
  const inputEl = el('username-input');
  const errEl   = el('login-error');
  const username = inputEl?.value?.trim() ?? '';

  // Validasi input
  if (!username) {
    if (errEl) errEl.textContent = '⚠️ Nama tidak boleh kosong.';
    inputEl?.focus();
    return;
  }
  if (username.length < 2) {
    if (errEl) errEl.textContent = '⚠️ Nama minimal 2 karakter.';
    return;
  }
  if (username.length > 50) {
    if (errEl) errEl.textContent = '⚠️ Nama maksimal 50 karakter.';
    return;
  }

  // Simpan dan masuk
  state.username = username;
  state.mode     = 'olivia';
  localStorage.setItem('olivia_user_login', username);

  _tampilkanGame();
}

/** Handle klik tombol logout */
function _handleLogout() {
  if (state.isProcessing) return; // Cegah logout saat AI sedang berpikir
  if (!confirm('Yakin ingin keluar dari kamar Olivia?')) return;

  // Reset state
  state.username      = null;
  state.mode          = 'olivia';
  state.isProcessing  = false;
  state.emosiTerakhir = 'netral';
  localStorage.removeItem('olivia_user_login');

  // Bersihkan input login
  const inputEl = el('username-input');
  if (inputEl) inputEl.value = '';

  // Hapus event listener resize
  window.removeEventListener('resize', () => _scrollChatKeBawah());

  _tampilkanLogin();
}

// ================================================================
// HUD UPDATE
// ================================================================

/** Update elemen-elemen HUD (nama user, avatar) */
function _updateHUD() {
  const { username } = state;
  if (!username) return;

  // Nama di HUD
  const labelEl = el('user-label');
  if (labelEl) labelEl.textContent = username;

  // Avatar (inisial huruf pertama)
  const avatarEl = el('user-avatar');
  if (avatarEl) {
    avatarEl.textContent = username.charAt(0).toUpperCase();
  }

  // Update label header chat
  _updateChatHeaderLabel();
}

/** Update teks label di chat panel header */
function _updateChatHeaderLabel() {
  const labelEl = el('chat-header-label');
  if (!labelEl) return;

  labelEl.textContent = state.mode === 'olivia'
    ? '💕 Log Interaksi — Olivia'
    : '🔧 Log Interaksi — AIRA Support';
}

// ================================================================
// STATUS BAR
// ================================================================

/**
 * Update status indicator di HUD.
 *
 * @param {"idle"|"loading"|"thinking"|"error"|"offline"} type
 * @param {string} [customText] - Teks override (opsional)
 */
function _setStatus(type, customText = '') {
  const dotEl  = el('status-dot');
  const textEl = el('status-text');

  const STATUS = {
    idle:     { cls: 'active',  label: '🟢 Siap' },
    loading:  { cls: 'pending', label: '⏳ Memuat AI...' },
    thinking: { cls: 'pending', label: '🤔 Berpikir...' },
    error:    { cls: 'error',   label: '❌ Error' },
    offline:  { cls: 'offline', label: '⚫ Mode Offline' },
  };

  const cfg = STATUS[type] ?? STATUS.idle;

  if (dotEl) {
    dotEl.className = `status-dot ${cfg.cls}`;
  }
  if (textEl) {
    textEl.textContent = customText || cfg.label;
  }
}

// ================================================================
// MODE SWITCHING
// ================================================================

/**
 * Pindah ke mode agent yang berbeda.
 * Memblokir switching saat AI sedang memproses.
 *
 * @param {"olivia"|"aira_acs"} newMode
 */
function _switchMode(newMode) {
  if (state.mode === newMode) return;
  if (state.isProcessing) return; // Jangan switch saat AI berpikir

  state.mode = newMode;

  // Update tab UI
  const tabOlivia = el('tab-olivia');
  const tabAira   = el('tab-aira');

  if (tabOlivia) {
    tabOlivia.classList.toggle('active', newMode === 'olivia');
    tabOlivia.setAttribute('aria-pressed', String(newMode === 'olivia'));
  }
  if (tabAira) {
    tabAira.classList.toggle('active', newMode === 'aira_acs');
    tabAira.setAttribute('aria-pressed', String(newMode === 'aira_acs'));
  }

  // Update karakter visual
  _updateCharacterVisual('netral');
  _updateChatHeaderLabel();
  _renderChat();

  // Focus ke input
  el('user-input')?.focus();
}

// ================================================================
// EXPRESSION CONTROLLER
// ================================================================

const expressionState = {
  currentEmotion: 'netral',
  isBlinking: false,
  thinkingFrame: 1,
  blinkTimer: null,
  thinkingTimer: null,
};

// ================================================================
// CHARACTER VISUAL (Emotion Feedback)
// ================================================================

/**
 * Update tampilan karakter berdasarkan emosi.
 * Mengganti src gambar dan efek CSS filter.
 *
 * @param {"netral"|"sedih"|"marah"|"bahagia"|"cemas"|"thinking"} emosi
 */
function _updateCharacterVisual(emosi) {
  const imgEl      = el('character-img');
  const fallbackEl = el('character-fallback');
  const stageEl    = document.querySelector('.character-stage');

  if (!imgEl && !fallbackEl) return;

  expressionState.currentEmotion = emosi;

  // // Map emosi ke nama file aset
  // const ASSET_MAP = {
  //   netral:   './assets/olivia-normal.png',
  //   bahagia:  './assets/olivia-happy.png',
  //   sedih:    './assets/olivia-sad.png',
  //   marah:    './assets/olivia-concerned.png',
  //   cemas:    './assets/olivia-concerned.png',
  //   thinking: './assets/olivia-thinking.png',
  // };

  // Map emosi ke class CSS efek glow
  const FILTER_MAP = {
    netral:   '',
    bahagia:  'emotion-happy',
    sedih:    'emotion-sad',
    marah:    'emotion-angry',
    cemas:    'emotion-sad',
    thinking: 'emotion-thinking',
  };

  // // Update src gambar
  // if (imgEl) {
  //   const newSrc = ASSET_MAP[emosi] ?? ASSET_MAP.netral;

  //   // Hanya update jika berbeda untuk mencegah flicker
  //   if (imgEl.src !== newSrc) {
  //     imgEl.src = newSrc;
  //   }

  //   // Update class efek visual
  //   imgEl.className = 'character-img'; // reset
  //   const filterCls = FILTER_MAP[emosi] ?? '';
  //   if (filterCls) imgEl.classList.add(filterCls);
  // }

  // Update src gambar
  if (imgEl) {

    // gunakan sistem gambar baru
    _applyCharacterImage();

    // tetap pertahankan brightness / glow effect
    imgEl.className = 'character-img';

    const filterCls = FILTER_MAP[emosi] ?? '';
    if (filterCls) {
      imgEl.classList.add(filterCls);
    }
  }

  // Update emoji fallback berdasarkan emosi
  if (fallbackEl) {
    const EMOJI_MAP = {
      netral:   '🌸',
      bahagia:  '💕',
      sedih:    '🌙',
      marah:    '😔',
      cemas:    '😟',
      thinking: '💭',
    };
    fallbackEl.textContent = EMOJI_MAP[emosi] ?? '🌸';
  }

  // Untuk mode AIRA: tampilan berbeda
  if (state.mode === 'aira_acs') {
    // fallbackEl.textContent = '🔧';
    if (fallbackEl) {
      fallbackEl.textContent = '🔧';
    }
    // Tambahkan baris ini untuk mereset class glow filter Olivia pada image element
    if (imgEl) {
      imgEl.className = 'character-img'; 
    }
  }
}

function _applyCharacterImage() {
  const imgEl = el('character-img');
  if (!imgEl) return;

  const emosi = expressionState.currentEmotion;
const isBlinking = expressionState.isBlinking;
  let src = '';

  // ─── KONDISI 1: JIKA SEDANG DI MODE OLIVIA ───
  if (state.mode === 'olivia') {
    switch (emosi) {
      case 'bahagia':
        src = isBlinking ? './assets/olivia-happy-blink.png' : './assets/olivia-happy.png';
        break;
      case 'sedih':
        src = isBlinking ? './assets/olivia-sad-blink.png' : './assets/olivia-sad.png';
        break;
      case 'marah':
        src = isBlinking ? './assets/olivia-concerned-blink.png' : './assets/olivia-concerned.png';
        break;
      case 'cemas':
        src = isBlinking ? './assets/olivia-afraid-blink.png' : './assets/olivia-afraid.png';
        break;
      case 'thinking':
        if (expressionState.thinkingFrame === 1) {
          src = isBlinking ? './assets/olivia-thinking-1-blink.png' : './assets/olivia-thinking-1.png';
        } else {
          src = isBlinking ? './assets/olivia-thinking-2-blink.png' : './assets/olivia-thinking-2.png';
        }
        break;
      default:
        src = isBlinking ? './assets/olivia-normal-blink.png' : './assets/olivia-normal.png';
    }
  } 
  
  // ─── KONDISI 2: JIKA SEDANG DI MODE AIRA (SUPPORT SYSTEM) ───
  else if (state.mode === 'aira_acs') {
    switch (emosi) {
      case 'thinking':
        // Animasi berpikir AIRA (berganti frame 1 & 2 + bisa sambil kedip)
        if (expressionState.thinkingFrame === 1) {
          src = isBlinking ? './assets/aira-thinking-1-blink.png' : './assets/aira-thinking-1.png';
        } else {
          src = isBlinking ? './assets/aira-thinking-2-blink.png' : './assets/aira-thinking-2.png';
        }
        break;
      default:
        // Mode santai / siap menerima chat dari user untuk AIRA
        src = isBlinking ? './assets/aira-normal-blink.png' : './assets/aira-normal.png';
    }
  }

  // cegah reload gambar yang sama
  const currentFile = imgEl.src.split('/').pop();

  if (currentFile !== src.split('/').pop()) {
    imgEl.src = src;
  }
}

function _startBlinkSystem() {
  function scheduleBlink() {

    const delay = 2000 + Math.random() * 4000;
    // 2 - 6 detik

    expressionState.blinkTimer = setTimeout(() => {

      expressionState.isBlinking = true;
      _applyCharacterImage();

      setTimeout(() => {
        expressionState.isBlinking = false;
        _applyCharacterImage();

        scheduleBlink();
      }, 150);

    }, delay);
  }

  scheduleBlink();
}

function _startThinkingAnimation() {

  expressionState.thinkingTimer = setInterval(() => {

    if (expressionState.currentEmotion !== 'thinking')
      return;

    expressionState.thinkingFrame =
      Math.random() < 0.5 ? 1 : 2;

    _applyCharacterImage();

  }, 1000);
}

// ================================================================
// CHAT RENDER
// ================================================================

/**
 * Render ulang seluruh isi chat box dari localStorage.
 * Dipanggil setiap kali riwayat chat berubah atau mode berganti.
 */
function _renderChat() {
  const chatBoxEl = el('chat-box');
  if (!chatBoxEl) return;

  chatBoxEl.innerHTML = '';

  if (!state.username) return;

  const riwayat = dapatkanRiwayatChat(state.username, state.mode);

  // Tampilkan empty state jika belum ada pesan
  if (riwayat.length === 0) {
    chatBoxEl.innerHTML = `
      <div class="chat-empty">
        <span class="empty-icon" aria-hidden="true">
          ${state.mode === 'olivia' ? '💕' : '🔧'}
        </span>
        <p>
          ${state.mode === 'olivia'
            ? `Hei <strong>${_escapeHtml(state.username)}</strong>! Sapa Olivia untuk memulai percakapan~`
            : 'Ceritakan masalah teknismu kepada AIRA. Siap membantu 24/7!'}
        </p>
        <p style="font-size: 0.8rem; opacity: 0.6; margin-top: 4px;">
          ${state.mode === 'aira_acs'
            ? 'Contoh: "Game saya crash terus" atau "Tidak ada suara"'
            : 'Contoh: "Hei Olivia, apa kabar?" atau "Aku lagi sedih nih..."'}
        </p>
      </div>
    `;
    return;
  }

  // Render setiap pesan
  const fragment = document.createDocumentFragment();
  for (const pesan of riwayat) {
    const msgEl = _buatElemenPesan(pesan);
    fragment.appendChild(msgEl);
  }
  chatBoxEl.appendChild(fragment);

  // Scroll ke pesan terbaru
  _scrollChatKeBawah();
}

/**
 * Buat elemen DOM untuk satu pesan (user atau AI).
 *
 * @param {{ role: string, message: string, time: string }} pesan
 * @returns {HTMLElement}
 */
function _buatElemenPesan(pesan) {
  const isUser = pesan.role === 'user';

  const row = document.createElement('div');
  row.className = `msg-row ${isUser ? 'user' : 'ai'}`;

  // Tentukan nama pengirim dan class berdasarkan mode dan role
  const senderClass = isUser
    ? 'user'
    : state.mode === 'olivia' ? 'olivia' : 'aira';

  const senderLabel = isUser
    ? _escapeHtml(state.username ?? 'Kamu')
    : state.mode === 'olivia' ? 'Olivia 💕' : 'AIRA 🔧';

  const bubbleClass = isUser
    ? 'chat-bubble-user'
    : state.mode === 'olivia' ? 'chat-bubble-olivia' : 'chat-bubble-aira';

  // Escape untuk keamanan XSS
  const safeMsg = _formatPesan(pesan.message, !isUser && state.mode === 'aira_acs');

  row.innerHTML = `
    <div class="msg-meta">
      <span class="msg-sender ${senderClass}">${senderLabel}</span>
      <span class="msg-time">${_escapeHtml(pesan.time ?? '')}</span>
    </div>
    <div class="msg-bubble">
      <div class="${bubbleClass}">${safeMsg}</div>
    </div>
  `;

  return row;
}

/**
 * Format teks pesan — escape HTML, dan untuk AIRA: ubah newline ke <br>
 * dan format nomor list menjadi elemen list yang lebih rapi.
 *
 * @param {string}  teks
 * @param {boolean} isAira - Apakah ini pesan AIRA (format list lebih kaya)
 * @returns {string}       - HTML string aman
 */
function _formatPesan(teks, isAira = false) {
  // Escape HTML dasar
  const safe = _escapeHtml(teks);

  if (!isAira) {
    // Untuk Olivia: hanya ubah newline ke <br>
    return safe.replace(/\n/g, '<br>');
  }

  // Untuk AIRA: format numbered list lebih rapi
  const baris = safe.split('\n');
  const hasil = [];
  let inList   = false;

  for (const b of baris) {
    const trimmed = b.trim();

    // Deteksi pola "1. " atau "2. " (numbered list)
    const listMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    if (listMatch) {
      if (!inList) {
        hasil.push('<ol style="list-style:decimal; padding-left: 1.2em; margin: 4px 0;">');
        inList = true;
      }
      hasil.push(`<li style="margin-bottom: 4px;">${listMatch[2]}</li>`);
    } else {
      if (inList) {
        hasil.push('</ol>');
        inList = false;
      }
      if (trimmed === '' || trimmed === '---') {
        hasil.push('<br>');
      } else if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
        // Section header — bold
        hasil.push(`<strong style="display: block; margin: 6px 0 2px; color: #6ee7b7;">${trimmed}</strong>`);
      } else {
        hasil.push(`${trimmed}<br>`);
      }
    }
  }

  if (inList) hasil.push('</ol>');

  return hasil.join('');
}

/**
 * Tambahkan bubble "berpikir" sementara menunggu respons AI.
 */
function _tambahThinkingBubble() {
  const chatBoxEl = el('chat-box');
  if (!chatBoxEl) return;

  // Hapus empty state jika ada
  chatBoxEl.querySelector('.chat-empty')?.remove();

  const row = document.createElement('div');
  row.className = 'msg-row ai';
  row.id        = 'thinking-bubble';

  const senderLabel = state.mode === 'olivia' ? 'Olivia 💕' : 'AIRA 🔧';
  const senderClass = state.mode === 'olivia' ? 'olivia' : 'aira';
  const dotsClass   = state.mode === 'olivia' ? 'loading-dots' : 'loading-dots dots-aira';

  row.innerHTML = `
    <div class="msg-meta">
      <span class="msg-sender ${senderClass}">${senderLabel}</span>
    </div>
    <div class="msg-bubble">
      <div class="thinking-bubble" aria-label="Sedang berpikir">
        <div class="${dotsClass}" aria-hidden="true">
          <span></span><span></span><span></span>
        </div>
        <span style="font-size: 0.75rem; opacity: 0.7; margin-left: 4px;">
          ${state.mode === 'olivia' ? 'sedang mengetik...' : 'mencari solusi...'}
        </span>
      </div>
    </div>
  `;

  chatBoxEl.appendChild(row);
  _scrollChatKeBawah();
}

/** Hapus bubble thinking dari DOM */
function _hapusThinkingBubble() {
  el('thinking-bubble')?.remove();
}

/** Scroll chat ke pesan paling bawah dengan smooth behavior */
function _scrollChatKeBawah() {
  const chatBoxEl = el('chat-box');
  if (chatBoxEl) {
    chatBoxEl.scrollTop = chatBoxEl.scrollHeight;
  }
}

// ================================================================
// SEND MESSAGE
// ================================================================

/** Handle pengiriman pesan dari user */
async function _handleSendMessage() {
  // Cegah double-send
  if (state.isProcessing) return;

  const inputEl  = el('user-input');
  const teksUser = inputEl?.value?.trim() ?? '';

  if (!teksUser || !state.username) return;

  // ── 1. Lock UI ──
  state.isProcessing = true;
  if (inputEl) { inputEl.value = ''; inputEl.disabled = true; }
  const btnSend = el('btn-send');
  if (btnSend) btnSend.disabled = true;

  // ── 2. Deteksi emosi dari pesan user ──
  const emosi = deteksiEmosi(teksUser);
  state.emosiTerakhir = emosi;

  // Update visual karakter ke mode "thinking"
  _updateCharacterVisual('thinking');

  // Variabel bantuan untuk memanipulasi DOM streaming secara real-time
  let streamingRow = null;
  let streamingBubbleInner = null;
  let accumulatedText = "";
  
  try {
    // ── 3. Muat AI Engine jika belum siap ──
    if (!isAIReady()) {
      show('loading-overlay');
      _setStatus('loading');

      await loadAIEngine((statusMsg) => {
        const loadTextEl = el('loading-status-text');
        if (loadTextEl) loadTextEl.textContent = statusMsg;
        _setStatus('loading', statusMsg);
      });

      state.aiLoaded = true;
      hide('loading-overlay');
    }

    // ── 4. Render pesan user ──
    simpanPesanChat(state.username, state.mode, 'user', teksUser);
    _renderChat();

    // ── 5. Tampilkan thinking bubble ──
    _tambahThinkingBubble();
    _setStatus('thinking');

    // ── 6. Generate respons AI ──
    const respons = await prosesChatLokal(state.username, state.mode, teksUser, (token) => {
      accumulatedText += token;

      // Logika khusus streaming hanya untuk mode AIRA Support
      if (state.mode === 'aira_acs') {
        const thinkingBubble = el('thinking-bubble');
        const chatBoxEl = el('chat-box');

        // Jika bubble streaming belum dibuat di DOM, buat pembungkusnya sekali saja
        if (!streamingRow && chatBoxEl) {
          streamingRow = document.createElement('div');
          streamingRow.className = 'msg-row ai';
          
          // Menggunakan inline style warna abu-abu (clr-text-secondary) untuk menyamarkan teks awal
          streamingRow.innerHTML = `
            <div class="msg-meta">
              <span class="msg-sender aira">AIRA 🔧</span>
              <span class="msg-time">${new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })}</span>
            </div>
            <div class="msg-bubble">
              <div class="chat-bubble-aira" id="aira-streaming-content" style="color: var(--clr-text-secondary) !important; border-color: var(--clr-border-subtle) !important; filter: none !important;"></div>
            </div>
          `;

          // Trik penting: Sisipkan SEBELUM thinking bubble agar loading tetap berada paling bawah
          if (thinkingBubble) {
            chatBoxEl.insertBefore(streamingRow, thinkingBubble);
          } else {
            chatBoxEl.appendChild(streamingRow);
          }
          
          streamingBubbleInner = el('aira-streaming-content');
        }

        // Tampilkan teks akumulasi secara real-time ke dalam bubble abu-abu
        if (streamingBubbleInner) {
          // Melakukan pencegahan XSS dasar & mengubah baris baru (\n) ke break tag (<br>) saat proses streaming
          streamingBubbleInner.innerHTML = _escapeHtml(accumulatedText).replace(/\n/g, '<br>');
          _scrollChatKeBawah();
        }
      }
    });

    // ── 7. Hapus thinking bubble dan render ulang dengan respons ──
    _hapusThinkingBubble();

    if (state.mode === 'aira_acs' && streamingBubbleInner) {
      // Ubah warna teks menjadi normal (menghapus paksaan style warna abu-abu agar kembali ke warna hijau asli #6ee7b7)
      streamingBubbleInner.style.color = '';
      streamingBubbleInner.style.borderColor = '';
      
      // Terapkan Rich Text Formatter (mengubah list nomor beralih menjadi tag <ol> dan <li> yang rapi)
      streamingBubbleInner.innerHTML = _formatPesan(respons, true);
      _scrollChatKeBawah();
    } else {
      // Untuk mode Olivia, biarkan menggunakan animasi render standar bawaanmu
      _renderChat();
    }

    // ── 8. Update karakter visual berdasarkan emosi ──
    _updateCharacterVisual(emosi);
    _setStatus('idle');

  } catch (err) {
    console.error('[Main] Error saat prosesChatLokal:', err);
    _hapusThinkingBubble();
    hide('loading-overlay');
    _setStatus('error', `❌ ${err.message ?? 'Kesalahan tidak diketahui'}`);
    _updateCharacterVisual('netral');

    // Jika terjadi crash di tengah jalan saat streaming, hapus baris rusak yang menggantung
    if (streamingRow) streamingRow.remove();

    // Tampilkan error di dalam chat sebagai bubble khusus
    _tampilkanPesanError(err.message ?? 'Terjadi kesalahan. Silakan coba lagi.');

    // Auto-reset status setelah 5 detik
    setTimeout(() => _setStatus('idle'), 5000);

  } finally {
    // ── Unlock UI ──
    state.isProcessing = false;
    if (inputEl)  { inputEl.disabled  = false; inputEl.focus(); }
    if (btnSend)  { btnSend.disabled  = false; }
  }
}

/**
 * Tampilkan pesan error di dalam chat box.
 * @param {string} pesanError
 */
function _tampilkanPesanError(pesanError) {
  const chatBoxEl = el('chat-box');
  if (!chatBoxEl) return;

  const errRow = document.createElement('div');
  errRow.className = 'msg-row ai';
  errRow.innerHTML = `
    <div class="msg-bubble">
      <div class="chat-bubble-aira" style="background: rgba(239,35,60,0.1); border-color: rgba(239,35,60,0.3); color: #ff8fa3;">
        ⚠️ <strong>Sistem:</strong> ${_escapeHtml(pesanError)}
        <br><span style="font-size: 0.75rem; opacity: 0.7;">Coba kirim ulang atau muat ulang halaman.</span>
      </div>
    </div>
  `;

  chatBoxEl.appendChild(errRow);
  _scrollChatKeBawah();
}

// ================================================================
// CLEAR CHAT
// ================================================================

/** Handle hapus riwayat chat untuk mode aktif */
function _handleClearChat() {
  if (!state.username) return;
  if (state.isProcessing) return;

  const modeName = state.mode === 'olivia' ? 'Olivia' : 'AIRA ACS';
  if (!confirm(`Hapus seluruh riwayat percakapan dengan ${modeName}?\n\nTindakan ini tidak dapat dibatalkan.`)) return;

  hapusRiwayatChat(state.username, state.mode);
  state.emosiTerakhir = 'netral';
  _updateCharacterVisual('netral');
  _renderChat();
  el('user-input')?.focus();
}

// ================================================================
// EVENT LISTENERS
// ================================================================

/** Pasang semua event listener DOM */
function _pasangSemuaEventListener() {
  // ── LOGIN ──
  el('btn-login')?.addEventListener('click', _handleLogin);

  el('username-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') _handleLogin();
  });

  // Hapus pesan error saat user mulai mengetik
  el('username-input')?.addEventListener('input', () => {
    const errEl = el('login-error');
    if (errEl) errEl.textContent = '';
  });

  // ── LOGOUT ──
  el('btn-logout')?.addEventListener('click', _handleLogout);

  // ── MODE TABS ──
  el('tab-olivia')?.addEventListener('click', () => _switchMode('olivia'));
  el('tab-aira')?.addEventListener('click',   () => _switchMode('aira_acs'));

  // ── CHAT INPUT & SEND ──
  el('user-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      _handleSendMessage();
    }
  });

  el('btn-send')?.addEventListener('click', _handleSendMessage);

  // ── CLEAR CHAT ──
  el('btn-clear-chat')?.addEventListener('click', _handleClearChat);

  // ── LOADING OVERLAY (klik untuk dismiss jika sudah selesai) ──
  el('loading-overlay')?.addEventListener('click', (e) => {
    // Hanya dismiss jika AI sudah siap
    if (isAIReady() && e.target === el('loading-overlay')) {
      hide('loading-overlay');
    }
  });
}

/** Pasang keyboard shortcuts global */
function _pasangKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + 1 → Chat Olivia
    if ((e.ctrlKey || e.metaKey) && e.key === '1') {
      e.preventDefault();
      if (!isHidden('game-section')) _switchMode('olivia');
    }

    // Ctrl/Cmd + 2 → AIRA Support
    if ((e.ctrlKey || e.metaKey) && e.key === '2') {
      e.preventDefault();
      if (!isHidden('game-section')) _switchMode('aira_acs');
    }

    // Escape → Bersihkan error login
    if (e.key === 'Escape') {
      const errEl = el('login-error');
      if (errEl) errEl.textContent = '';
    }
  });
}

// ================================================================
// SECURITY UTILITIES
// ================================================================

/**
 * Escape string menjadi HTML aman (cegah XSS injection).
 * @param {string} str
 * @returns {string}
 */
function _escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}
