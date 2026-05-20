import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from config_rl import NUM_EPISODES, TIMESTEPS_PER_EPISODE

RESULT_DIR = "results"

def get_latest_file(pattern):
    """Fungsi pembantu untuk mencari file terbaru berdasarkan pola nama."""
    search_path = os.path.join(RESULT_DIR, pattern)
    list_of_files = glob.glob(search_path)
    
    if not list_of_files:
        return None
        
    # Mengembalikan file dengan waktu modifikasi (mtime) paling baru
    return max(list_of_files, key=os.path.getmtime)

def main():
    file_rl = get_latest_file("hasil_evaluasi_wsn_dqn_E1000_T500_*.csv")
    file_random = get_latest_file("hasil_baseline_random_E1000_T500_*.csv")
#--- UBAH NAMA CSV SESUAI NAMA MODEL---#
    if not file_rl or not file_random:
        print("[ERROR] File data tidak lengkap di folder 'results'.")
        print(f"File DQN ditemukan: {file_rl is not None}")
        print(f"File Baseline ditemukan: {file_random is not None}")
        print("Pastikan script extract_logs.py dan baseline_random.py sudah dijalankan.")
        return

    print(f"[INFO] Membaca file DQN      : {os.path.basename(file_rl)}")
    print(f"[INFO] Membaca file Baseline : {os.path.basename(file_random)}")

    # 2. Baca Data
    try:
        df_rl = pd.read_csv(file_rl)
        df_rand = pd.read_csv(file_random)
        print("[INFO] Data berhasil dimuat.")
    except Exception as e:
        print(f"[ERROR] Gagal membaca CSV: {e}")
        return

    # 3. Penyesuaian Sumbu X (Time-steps)
    # Data RL biasanya sudah punya kolom 'step'.
    # Data Random punya 'episode', kita konversi jadi 'step' agar sejajar.
    if 'step' not in df_rand.columns:
        # Asumsi: Setiap episode random menghabiskan TIMESTEPS_PER_EPISODE
        df_rand['step'] = df_rand['episode'] * TIMESTEPS_PER_EPISODE

    # 4. Plotting
    plt.style.use('ggplot') # Gaya grafik professional
    plt.figure(figsize=(10, 6))

     # --- Plot Garis DQN (Usulan) ---
    # Kita gunakan garis tegas biru
    plt.plot(df_rl['step'], df_rl['reward_mean'], 
             color='purple', linewidth=1, 
             label='Proposed Method (DQN)')

    # --- Plot Garis Random (Baseline) ---
    # Kita gunakan garis putus-putus merah
    plt.plot(df_rand['step'], df_rand['reward_sma'], 
             color='yellow', linewidth=1.5, 
             label='Baseline (Random)')


    # 5. Dekorasi Grafik
    plt.title('Perbandingan Kinerja: DQN Agent vs Random Baseline', fontsize=14)
    plt.xlabel('Langkah Simulasi (Time-steps)', fontsize=12)
    plt.ylabel('Rata-rata Reward', fontsize=12)
    plt.legend(loc='lower right') # Posisi legenda
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 6. Simpan Hasil
    name_rl = os.path.basename(file_rl).replace('.csv', '')
    name_rand = os.path.basename(file_random).replace('.csv', '')
    # Gabungkan menjadi nama output grafik
    output_filename = f"grafik_perbandingan_{name_rl}_VS_{name_rand}.png"
    output_img = os.path.join(RESULT_DIR, output_filename)
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"[SUKSES] Grafik perbandingan disimpan: {output_img}")
    
    # Tampilkan pesan analisis singkat
    avg_rl = df_rl['reward_mean'].iloc[-10:].mean()
    avg_rand = df_rand['reward_sma'].iloc[-10:].mean()
    print(f"\n[ANALISIS SINGKAT]")
    print(f"Rata-rata Reward Akhir DQN   : {avg_rl:.2f}")
    print(f"Rata-rata Reward Akhir Random: {avg_rand:.2f}")
    
    if avg_rl > avg_rand:
        print("Kesimpulan: Metode DQN LEBIH BAIK daripada Random.")
    else:
        print("Kesimpulan: Metode DQN BELUM optimal (Cek training lebih lama).")

if __name__ == "__main__":
    main()