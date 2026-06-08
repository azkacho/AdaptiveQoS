"Plotting script untuk visualisasi hasil training "
"dari file CSV terbaru yang dihasilkan oleh main-rl.py."
"dapat diwakili oleh main-rl.py ketika sudah selesai training model"
"hanya untuk evaluasi plot secara lebih jelas tanpa harus menjalankan ulang training."

import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# ── 1. SETUP PATH DINAMIS ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# ── 2. DETEKSI FILE CSV TERBARU ─────────────────────────────────────────
# Menggunakan wildcard (*) agar bisa membaca model apa pun yang terbaru
log_pattern = os.path.join(LOGS_DIR, "train_log_wsn_dqn_E2000_T1500*.csv")
list_of_files = glob.glob(log_pattern)

if not list_of_files:
    print(f"[ERROR] Tidak ditemukan file di: {log_pattern}")
    print("Pastikan folder 'logs' sudah berisi file .csv hasil running main-rl.py")
else:
    # Mengambil file yang paling baru berdasarkan waktu modifikasi
    latest_file = max(list_of_files, key=os.path.getmtime)
    print(f"[INFO] Membaca data dari file: {latest_file}")

    # Mengambil nama asli file CSV (contoh: train_log_wsn_dqn_E500_T1000_20260525)
    filename_only = os.path.basename(latest_file).replace(".csv", "")

    # ── 3. MEMBACA DATA MENGGUNAKAN PANDAS ──────────────────────────────
    df = pd.read_csv(latest_file)

    # Memastikan kolom tersedia (sesuai header di CSV Anda)
    episodes = df['Episode']
    rewards = df['Reward']
    qos_scores = df['Avg_QoS']
    drops = df['Drops']
    retries = df['Retries']

    # ── 4. PLOTTING ─────────────────────────────────────────────────────
    plt.figure(figsize=(15, 12))

    # Subplot 1: Reward
    plt.subplot(4, 1, 1)
    plt.plot(episodes, rewards, color='b', label='Reward')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title('Training Rewards')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 2: QoS
    plt.subplot(4, 1, 2)
    plt.plot(episodes, qos_scores, color='g', label='Avg QoS')
    plt.title('Average QoS Score (1.0 = Perfect)')
    plt.ylabel('QoS Score')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 3: Drops
    plt.subplot(4, 1, 3)
    plt.plot(episodes, drops, color='r', label='Drops')
    plt.title('Connection Drops')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 4: Retries
    plt.subplot(4, 1, 4)
    plt.plot(episodes, retries, color='orange', label='Retries')
    plt.title('Total Retransmissions')
    plt.ylabel('Count')
    plt.xlabel('Episode')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    
    # ── 5. PENAMAAN DAN PENYIMPANAN OUTPUT DINAMIS ──────────────────────
    # Mengganti kata awalan CSV untuk menjadi nama file gambar (PNG)
    # Contoh: train_log_wsn_dqn_E500_T1000_... -> training_metrics_E500_T1000_...
    if "train_log_wsn_dqn_" in filename_only:
        output_filename = filename_only.replace("train_log_wsn_dqn_", "training_metrics_") + ".png"
    elif "train_log_" in filename_only:
        output_filename = filename_only.replace("train_log_", "training_metrics_") + ".png"
    else:
        output_filename = f"training_metrics_{filename_only}.png"
        
    output_plot = os.path.join(RESULT_DIR, output_filename)
    
    plt.savefig(output_plot)
    print(f"[BERHASIL] Grafik disimpan sebagai {output_plot}")
    plt.show()