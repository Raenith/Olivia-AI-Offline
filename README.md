# OLIVIA AI — Virtual Companion Experience

![WebGPU](https://img.shields.io/badge/WebGPU-Powered-blueviolet?style=for-the-badge)
![100% Offline](https://img.shields.io/badge/100%25-Offline-success?style=for-the-badge)
![Privacy](https://img.shields.io/badge/Privacy-Absolute-ff007f?style=for-the-badge)
![Server Cost](https://img.shields.io/badge/Server%20Cost-Rp%200-cyan?style=for-the-badge)

![models](https://drive.google.com/file/d/1uQ426Bx7ZkQ-lbrgLL_UxUTQ_1J40_OK/view?usp=sharing)
![Demo](https://raenith.github.io/Olivia-AI-Offline/Olivia-AI-WEB/index.html)

OLIVIA AI adalah platform interaksi asisten dan teman virtual 3D/2D berbasis **Edge AI** yang berjalan sepenuhnya di sisi klien (*client-side*). Proyek ini memanfaatkan akselerasi hardware modern menggunakan **WebGPU** untuk menjalankan Large Language Model (LLM) secara lokal langsung di dalam browser tanpa memerlukan server eksternal, koneksi internet berkelanjutan, ataupun biaya API.

Dikembangkan sebagai solusi nyata untuk tantangan privasi data dan efisiensi biaya operasional infrastruktur AI dalam kompetisi **"Build Real AI Agent. Solve Real Problems" — Qhomemart**.

---

## 🚀 Fitur Unggulan

* 🛡️ **Keamanan Privasi Absolut:** 0MB data dikirim ke cloud. Semua proses inferensi teks, analisis sentimen, dan memori percakapan diproses dan menetap selamanya di perangkat pengguna.
* ⚡ **Akselerasi WebGPU:** Menggunakan pipeline WebGPU via ONNX Runtime Web untuk kecepatan generasi teks tingkat tinggi langsung di GPU browser.
* 🌐 **100% Operasional Offline:** Setelah pengunduhan model pertama (*initial load*), aplikasi dapat diakses penuh tanpa koneksi internet sama sekali.
* 💰 **Nol Biaya Operasional (Zero Server Cost):** Skalabilitas tanpa batas untuk developer karena beban komputasi didistribusikan langsung ke perangkat masing-masing pengguna.
* 🧠 **Sistem Multi-Agent Dinamis:** Menyediakan dua persona AI interaktif yang saling melengkapi sesuai kebutuhan pengguna.

---

## 👥 Profil Agen AI

### 1. 🌸 Olivia Valen (Virtual Companion Mode)
Karakter AI dengan persona yang tenang, dewasa, dan empatik. Dilengkapi dengan **Local Emotion Detection Engine** untuk mendeteksi tanda-tanda stres atau sentimen negatif pada input pengguna dan merespons secara protektif serta menenangkan. Menggunakan teknik *sliding window context* untuk mengelola memori jangka pendek di `localStorage`.

### 2. 🔧 AIRA ACS (AI Customer Service Support Mode)
Asisten teknis cerdas yang bertugas memecahkan kendala teknis sistem atau gameplay secara logis. Menggunakan arsitektur **Local Keyword-Matching RAG (Retrieval-Augmented Generation)** dari basis pengetahuan JSON lokal untuk memberikan solusi terstruktur tanpa ketergantungan API pencarian cloud.

---

## 🛠️ Tech Stack & Arsitektur

* **Core AI Engine:** [Transformers.js v3](https://github.com/xenova/transformers.js) (Hugging Face)[cite: 2]
* **Model Bahasa:** Gemma4 E2B IT (Dioptimalkan ke format ONNX quantized)[cite: 2]
* **Runtime:** ONNX Runtime Web dengan WebGPU Execution Provider[cite: 2]
* **Penyimpanan Lokal:** `localStorage` (untuk Multi-turn Chat Memory) & `Cache Storage / IndexedDB` (untuk Model Weights Caching)[cite: 2]
* **Frontend:** Vanilla HTML5, CSS3 (Modern custom property design system), dan Asynchronous JavaScript[cite: 2].

---

## 📁 Struktur Direktori

```text
olivia-ai/
├── js/
│   ├── ai-engine.js
│   ├── knowledge.js
│   └── main.js
├── css/
│   └── style.css          # Desain sistem global, token warna, & layout dasar
├── assets/
│   ├── models/            # Models ai
│   ├── olivia-normal.png  # Aset visual karakter Olivia
│   └── colab.png          # Visual ilustrasi halaman tentang
├── index.html             # Halaman Utama (Landing Page)
├── private.html           # Halaman Kebijakan Privasi Lokal
├── game.html              # Aplikasi Utama (Kamar Virtual / Antarmuka Chat)
├── README.md              # Dokumentasi Proyek
└── .gitignore             # Pengaturan pengabaian repositori Git
```

# Olivia-AI-Offline
WebGPU-Powered Local Edge AI Virtual Companion &amp; Support Agent. 100% Offline, Zero Server Cost, and Absolute Privacy using Gemma4 E2B IT &amp; Transformers.js v3.
