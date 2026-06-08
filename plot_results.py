import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")

search_pattern = os.path.join(results_dir, "hasil_evaluasi_wsn_dqn_E2000_T1500_*.csv")
#--- UBAH NAMA CSV SESUAI NAMA MODEL---#
list_of_files = glob.glob(search_pattern)

if not list_of_files:
    print("[ERROR] Tidak ditemukan file hasil_evaluasi_*.csv di folder results.")
    print("Jalankan extract_logs.py terlebih dahulu.")
else:
    # Mengambil file yang paling baru dimodifikasi
    latest_file = max(list_of_files, key=os.path.getmtime)
    print(f"[INFO] Membaca data dari: {os.path.basename(latest_file)}")

    # 2. Ekstrak parameter menggunakan split
    # Asumsi nama file: hasil_evaluasi_wsn_dqn_E1000_T200_20260413_095650.csv
    filename_only = os.path.basename(latest_file).replace(".csv", "")
    parts = filename_only.split("_")
    
    # Mengambil elemen dari belakang berdasarkan posisi
    e_val = parts[-4]      # Mendapatkan "E1000"
    t_val = parts[-3]      # Mendapatkan "T200"
    timestamp = f"{parts[-2]}_{parts[-1]}" # Mendapatkan "20260413_095650"

    # 3. Membaca Data
    df = pd.read_csv(latest_file)
    
    # Mengatur ukuran figure (lebar, tinggi)
    fig, axs = plt.subplots(4, 2, figsize=(16, 22))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # 1. Grafik Reward WSN
    axs[0, 0].plot(df['step'], df['reward_mean'], color='blue')
    axs[0, 0].set_title('1. Grafik Reward WSN', fontweight='bold')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Grafik Episode Length
    axs[0, 1].plot(df['step'], df['episode_length'], color='green')
    axs[0, 1].set_title('2. Grafik Episode Length', fontweight='bold')
    axs[0, 1].set_ylabel('Steps')
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Grafik Lifetime WSN
    axs[1, 0].plot(df['step'], df['lifetime'], color='red')
    axs[1, 0].set_title('3. Grafik Lifetime WSN', fontweight='bold')
    axs[1, 0].set_ylabel('Lifetime Index')
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Grafik Konvergensi Reward (dengan SMA-10)
    axs[1, 1].plot(df['step'], df['reward_mean'], color='blue', alpha=0.3, label='Raw')
    sma10 = df['reward_mean'].rolling(window=10).mean()
    axs[1, 1].plot(df['step'], sma10, color='orange', linewidth=2, label='SMA-10')
    axs[1, 1].set_title('4. Konvergensi Reward (SMA-10)', fontweight='bold')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    # 5. Grafik Rata-rata QoS
    axs[2, 0].plot(df['step'], df['Avg_QoS'], color='purple')
    axs[2, 0].set_title('5. Grafik Rata-rata QoS', fontweight='bold')
    axs[2, 0].set_ylabel('QoS Score')
    axs[2, 0].set_ylim(0, 1.1)
    axs[2, 0].grid(True, alpha=0.3)

    # 6. Grafik Jumlah Kegagalan Koneksi (Drops)
    axs[2, 1].fill_between(df['step'], df['Drops'], color='brown', alpha=0.4)
    axs[2, 1].plot(df['step'], df['Drops'], color='brown')
    axs[2, 1].set_title('6. Jumlah Kegagalan Koneksi (Drops)', fontweight='bold')
    axs[2, 1].set_ylabel('Drops Count')
    axs[2, 1].grid(True, alpha=0.3)

    if 'EDF' in df.columns:
        axs[3, 0].scatter(df['step'], df['EDF'], s=10, color='darkcyan', alpha=0.6)
        axs[3, 0].set_title('7. EDF per Episode', fontweight='bold')
        axs[3, 0].set_ylabel('Deadline Value')
        axs[3, 0].grid(True, alpha=0.3)
    else:
        axs[3, 0].text(0.5, 0.5, 'Data EDF Tidak Tersedia', ha='center')

    
    if 'Retries' in df.columns:
        axs[3, 1].plot(df['step'], df['Retries'], color='orange')
        axs[3, 1].set_title('8. Total Retransmissions', fontweight='bold')
    else:
        axs[3, 1].axis('off')

    # Finalisasi layout
    fig.suptitle('Dashboard Evaluasi Adaptif QoS WSN-IoT', fontsize=18, fontweight='bold', y=0.95)
    
    output_filename = f"evaluasi_wsn_{e_val}_{t_val}_{timestamp}.png"
    output_png = os.path.join(results_dir, output_filename)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"[BERHASIL] File disimpan di: {output_png}")
    plt.show()