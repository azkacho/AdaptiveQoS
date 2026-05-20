"Plotting script untuk visualisasi hasil training "
"dari file CSV terbaru yang dihasilkan oleh main-rl.py."
"dapat diwakili oleh main-rl.py ketika sudah selesai training model"
"hanya untuk evaluasi plot secara lebih jelas tanpa harus menjalankan ulang training."


import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from datetime import datetime
from config_rl import NUM_EPISODES, TIMESTEPS_PER_EPISODE

# # --- 1. DETEKSI FILE CSV TERBARU ---
# # Mencari semua file yang berawalan 'training_log_' dan berakhiran '.csv'
# list_of_files = glob.glob('training_log_*.csv') --- direktori program lama
# # Mencari di folder AdaptiveQoS/logs/
# if not list_of_files:
#     # Jika tidak ketemu di folder utama, coba cari di folder results/
#     list_of_files = glob.glob('results/training_log_*.csv')

# if not list_of_files:
#     print("[ERROR] Tidak ditemukan file training_log_*.csv di direktori ini maupun di folder results.")
# else:
#     # Mengambil file yang paling baru berdasarkan waktu modifikasi
#     latest_file = max(list_of_files, key=os.path.getmtime)
#     print(f"[INFO] Membaca data dari file: {latest_file}")
# ----------------------------------------
RESULT_DIR = "results"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

log_pattern = os.path.join("logs", "train_log_wsn_dqn_E500_T1000*.csv")
#--- UBAH NAMA CSV SESUAI NAMA MODEL---#
list_of_files = glob.glob(log_pattern)

if not list_of_files:
    # Cek alternatif jika folder logs berada di level yang sama dengan script
    list_of_files = glob.glob("logs/training_log_*.csv")

if not list_of_files:
    print(f"[ERROR] Tidak ditemukan file training_log_*.csv di: {os.path.abspath('logs')}")
    print("Pastikan folder 'logs' sudah berisi file .csv hasil running main-rl.py")
else:
    # Mengambil file yang paling baru berdasarkan waktu modifikasi
    latest_file = max(list_of_files, key=os.path.getmtime)
    print(f"[INFO] Membaca data dari file: {latest_file}")

    filename_only = os.path.basename(latest_file).replace(".csv", "")
    parts = filename_only.split("_")
    timestamp = f"{parts[-2]}_{parts[-1]}"

    # --- 2. MEMBACA DATA MENGGUNAKAN PANDAS ---
    df = pd.read_csv(latest_file)

    # Memastikan kolom tersedia (sesuai header di CSV Anda)
    # Episode,Reward,Avg_QoS,Drops,Retries,Epsilon
    episodes = df['Episode']
    rewards = df['Reward']
    qos_scores = df['Avg_QoS']
    drops = df['Drops']
    retries = df['Retries']

    # --- 3. PLOTTING ---
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
    
    # Simpan hasil plot
    output_filename = f"training_metrics_E{NUM_EPISODES}_T{TIMESTEPS_PER_EPISODE}_{timestamp}.png"
    output_plot = os.path.join(RESULT_DIR, output_filename)
    
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig(output_plot)
    
    print(f"[BERHASIL] Grafik disimpan sebagai {output_plot}")
    plt.show()