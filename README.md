# AdaptiveQoS

# Folder Structure

AdaptiveQoS/
│
├── root/
│   ├── app.py                     ← Entry point Digital Twin (Dashboard Interface)
│   ├── wsn_rl_env.py              ← Environment simulasi WSN (Gymnasium Custom Env)
│   ├── main-rl.py                 ← Loop pelatihan agen DQN (PyTorch)
│   ├── config_rl.py               ← Konfigurasi parameter terpusat (Hyperparameters, Network, Sim)
│   │
│   ├── baseline_random.py         ← Agen baseline menggunakan aksi acak (untuk perbandingan)
│   ├── compare_results.py         ← Skrip analisis: Perbandingan metrik DQN vs Random
│   │
│   ├── extract_logs.py            ← Utilitas ekstraksi data dari TensorBoard event files
│   ├── find_central_node.py       ← Identifikasi target 'central node' untuk *Skenario C*
│   ├── analysis_training.py       ← Analisis kurva konvergensi pelatihan (Reward/Loss)
│   │
│   ├── requirements.txt           ← Daftar dependensi Python (PyTorch, Gymnasium, Dash, dll.)
│   └── README.md                  ← Dokumentasi utama project
│
├── components/                    ← Modul pendukung untuk interface Digital Twin
│   ├── __init__.py
│   ├── callbacks.py               ← Logika interaktif dashboard (input/output)
│   ├── layout.py                  ← Definisi tata letak antarmuka (UI Design)
│   └── logic.py                   ← Logika inti: Scoring parent & keputusan routing dinamis
│
├── assets/                                                     ← File statis untuk antarmuka
│   └── style.css                                               ← Kustomisasi styling CSS (Dark Mode/Theme)
│
├── models/                                                     ← Penyimpanan model terlatih (*.pth file)
│   └── wsn_dqn_*nama model*.pth                                ← Contoh output model DQN yang sudah konvergen
│
├── logs/                                                       ← Data mentah hasil pelatihan
│   ├── runs/                                                   ← TensorBoard event files
│   └── training_log__wsn_dqn_*nama_model*.csv                  ← Log performa per episode
│
├── results/                                                    ← Output analisis dan evaluasi
│   ├── grafik_perbandingan.png                                 ← Grafik hasil perbandingan DQN vs Baseline
│   └── hasil_evaluasi_wsn_dqn_*nama_model*.csv                 ← Data kuantitatif evaluasi skenario
│
└── experiment_data/                                            ← Data topologi/skenario yang diekstrak dari Digital Twin
    ├── skenarioA/B/C_*mode_*_autodrainON/OFF_nama_model.csv    ← Contoh data Skenario A
    └── skenarioA/B/C_*mode_*_autodrainON/OFF_nama_model.txt