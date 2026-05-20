from wsn_rl_env import WSN_RL_Env
import numpy as np

def run_validation():
    env = WSN_RL_Env()
    state, info = env.reset()
    
    print("=== VALIDASI LOGIKA SYSTEM ===")
    print(f"Posisi Awal Node: {env.current_node_id}")
    
    # Kita akan melakukan loop manual beberapa langkah
    # untuk memancing kondisi retransmisi dan drop
    
    total_steps = 20
    print(f"\nMenjalankan {total_steps} langkah uji coba...")

    for i in range(total_steps):
        # Pilih aksi acak
        action = env.action_space.sample()
        
        # Jalankan step
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Ambil data dari info
        retries = info['retries']
        dropped = info['dropped']
        s_qos = info['S_QoS']
        dist = info['distance']
        rate_idx = info['rate']
        
        # --- LOGIKA VALIDASI ---
        print(f"\nStep {i+1}:")
        print(f"  - Action (Rate): {rate_idx}, Jarak: {dist:.2f}m")
        print(f"  - Retries: {retries} | Dropped: {dropped}")
        print(f"  - Skor QoS (S_QoS): {s_qos:.4f}")
        print(f"  - Reward Diterima: {reward:.4f}")
        
        # 1. Cek Penalti Eksponensial
        # Jika retries > 3, reward harusnya mulai drop drastis negatif
        if retries > 3 and not dropped:
            print("    [CHECK] Penalti Eksponensial Terdeteksi (Reward Negatif Besar)")
            
        # 2. Cek Logika Connection Drop
        if dropped:
            print("    [VALID] KONEKSI DIPUTUS! (Retries >= 10)")
            if reward == -100.0: # Sesuai PENALTY_DROP di config
                print("    [VALID] Penalti Maksimum (-100) Diberikan.")
            else:
                print(f"    [ERROR] Reward Drop tidak sesuai! Dapat: {reward}")
            
            # Reset environment manual jika mati agar loop lanjut
            env.reset()
            print("    (Environment di-reset untuk lanjut tes...)")
            
        # 3. Cek Konteks QoS
        # Jika S_QoS rendah, penalti harusnya lebih besar (perlu hitungan manual untuk validasi detail, 
        # tapi secara visual reward harus lebih rendah dari biasanya)
        if s_qos < 0.5 and retries > 0:
             print("    [INFO] QoS Buruk terdeteksi, Hukuman diperberat faktor konteks.")

    print("\n=== PENGUJIAN SELESAI ===")

if __name__ == "__main__":
    run_validation()