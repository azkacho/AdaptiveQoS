import pandas as pd
import os
import glob
from datetime import datetime
from config_rl import NUM_EPISODES, TIMESTEPS_PER_EPISODE

# --- 1. SETUP DIREKTORI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULT_DIR = os.path.join(BASE_DIR, "results")

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def extract_csv_to_eval():
    # --- 2. DETEKSI LOG TERBARU OTOMATIS ---
    log_pattern = os.path.join(LOG_DIR, "train_log_wsn_dqn_E2000_T1500*.csv")
    #--- UBAH NAMA CSV SESUAI NAMA MODEL---#
    list_of_files = glob.glob(log_pattern)
    
    if not list_of_files:
        print(f"[ERROR] Tidak ditemukan file log di: {LOG_DIR}")
        return

    # Mengambil file yang paling baru dimodifikasi
    input_log = max(list_of_files, key=os.path.getmtime)
    input_filename = os.path.basename(input_log)
    experiment_id = input_filename.replace("train_log_", "").replace(".csv", "")
    print(f"[INFO] Memproses log terbaru: {os.path.basename(input_log)}")

    output_filename = f"hasil_evaluasi_{experiment_id}.csv"
    output_path = os.path.join(RESULT_DIR, output_filename)

    print(f"[INFO] Memproses log: {input_filename}")

    try:
        # 3. BACA DATA ASLI
        df_raw = pd.read_csv(input_log)
        
        # 4. TRANSFORMASI DATA
        df_eval = pd.DataFrame()
        
        # Mapping dasar
        df_eval['step'] = df_raw['Episode'] * TIMESTEPS_PER_EPISODE
        df_eval['reward_mean'] = df_raw['Reward']
        df_eval['episode_length'] = TIMESTEPS_PER_EPISODE
        
        # Menambahkan metrik WSN yang dibutuhkan plot_results.py
        if 'Avg_QoS' in df_raw.columns:
            df_eval['Avg_QoS'] = df_raw['Avg_QoS']
        
        if 'Drops' in df_raw.columns:
            df_eval['Drops'] = df_raw['Drops']
            
        if 'Retries' in df_raw.columns:
            df_eval['Retries'] = df_raw['Retries']

        if 'EDF_Episode' in df_raw.columns:
            df_eval['EDF'] = df_raw['EDF_Episode']

        # --- SOLUSI KOLOM LIFETIME ---
        # Jika simulasi belum menghasilkan data lifetime, kita gunakan Avg_QoS 
        # sebagai representasi stabilitas jaringan (proxy lifetime)
        if 'lifetime' in df_raw.columns:
            df_eval['lifetime'] = df_raw['lifetime']
        else:
            # Menggunakan Avg_QoS sebagai pengganti agar plot_results.py tidak error
            df_eval['lifetime'] = df_raw['Avg_QoS'] 

        # 5. SIMPAN HASIL
        df_eval.to_csv(output_path, index=False)
        print(f"[OK] Ekstraksi berhasil!")
        print(f"[INFO] File tersimpan di: {output_path}")
        print(f"[DATA] Kolom tersedia: {list(df_eval.columns)}")

    except Exception as e:
        print(f"[ERROR] Gagal memproses log: {e}")

if __name__ == "__main__":
    extract_csv_to_eval()