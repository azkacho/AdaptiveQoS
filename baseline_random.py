import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from wsn_rl_env import WSN_RL_Env
from datetime import datetime
from config_rl import NUM_EPISODES, TIMESTEPS_PER_EPISODE, RANDOM_SEED

def run_baseline():
    print("=== MEMULAI BASELINE (RANDOM AGENT) ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_suffix = f"E{NUM_EPISODES}_T{TIMESTEPS_PER_EPISODE}_{timestamp}"
    # 1. Inisialisasi Environment yang SAMA dengan DQN
    env = WSN_RL_Env()
    
    # List untuk menyimpan sejarah metrik
    history = {
        'rewards': [],
        'avg_qos': [],
        'drops': [],
        'retries': [],
        'edf': []
    }

    print(f"Running for {NUM_EPISODES} episodes...")

    for e in range(NUM_EPISODES):
        # Reset environment dengan seed yang sama agar kondisi awalnya adil
        state, _ = env.reset(seed=RANDOM_SEED + e)
        episode_edf = env.edf 
        total_reward = 0
        ep_qos_scores = []
        ep_retries = 0
        ep_drops = 0
        
        for time in range(TIMESTEPS_PER_EPISODE):
            # --- LOGIKA RANDOM AGENT ---
            # Memilih aksi secara acak dari ruang aksi yang tersedia
            action = env.action_space.sample()
            
            # Eksekusi step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Simpan data
            total_reward += reward
            ep_qos_scores.append(info.get('S_QoS', 0))
            ep_retries += info.get('retries', 0)
            if info.get('dropped', False):
                ep_drops += 1
            
            # Cek selesai
            if terminated or truncated:
                break
                
            state = next_state

        # Hitung rata-rata per episode
        avg_qos = np.mean(ep_qos_scores) if ep_qos_scores else 0
        
        # Simpan ke history
        history['rewards'].append(total_reward)
        history['avg_qos'].append(avg_qos)
        history['drops'].append(ep_drops)
        history['retries'].append(ep_retries)
        history['edf'].append(episode_edf)
        
        # Log progress setiap 50 episode (supaya tidak terlalu spam)
        if (e + 1) % 50 == 0:
            print(f"Baseline Episode: {e+1}/{NUM_EPISODES} | "
                  f"Avg Reward: {total_reward:.2f} | "
                  f"Drops: {ep_drops} | "
                  f"Retries: {ep_retries}"
                  f"EDF: {episode_edf:.4f}")
            

    # --- SIMPAN HASIL VISUALISASI BASELINE ---
    if not os.path.exists('results'):
        os.makedirs('results')

    fig, axs = plt.subplots(5, 1, figsize=(12, 18)) 
    
    # Plot 1: Reward
    axs[0].plot(history['rewards'], color='gray', alpha=0.7)
    axs[0].set_title('Baseline: Total Reward per Episode', fontweight='bold')
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: QoS
    axs[1].plot(history['avg_qos'], color='green', alpha=0.7)
    axs[1].set_title('Baseline: Average QoS Score', fontweight='bold')
    axs[1].set_ylim(0, 1.1)
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Retries
    axs[2].plot(history['retries'], color='orange', alpha=0.7)
    axs[2].set_title('Baseline: Total Retransmissions', fontweight='bold')
    axs[2].grid(True, alpha=0.3)
    
    # Plot 4: Drops
    axs[3].plot(history['drops'], color='red', alpha=0.7)
    axs[3].set_title('Baseline: Packet Drops', fontweight='bold')
    axs[3].grid(True, alpha=0.3)

    # Plot 5: EDF (Deadline Variation)
    axs[4].plot(history['edf'], color='purple', alpha=0.8)
    axs[4].set_title('Baseline: Deadline Variation (EDF) per Episode', fontweight='bold')
    axs[4].set_ylabel('EDF Value')
    axs[4].set_xlabel('Episode')
    axs[4].grid(True, alpha=0.3)
    
    # Mengatur jarak antar plot (pad=4.0) agar teks tidak saling menabrak
    plt.tight_layout(pad=4.0)
    plot_filename = f'baseline_metrics_{file_suffix}.png'
    plt.savefig(os.path.join('results', plot_filename))
    print(f"\n[SELESAI] Grafik Baseline disimpan ke results/{plot_filename}")
   # --- PROSES PENYIMPANAN KE CSV UNTUK KOMPARASI ---
    # Membuat DataFrame dari sejarah (history) yang terkumpul
    df_baseline = pd.DataFrame({
        'episode': range(len(history['rewards'])),
        'reward_mean': history['rewards'],
        # Menghitung reward_sma agar sesuai dengan yang dicari compare_results.py
        'reward_sma': pd.Series(history['rewards']).rolling(window=10, min_periods=1).mean(),
        'avg_qos': history['avg_qos'],
        'drops': history['drops'],
        'retries': history['retries'],
        'edf': history['edf']
    })

    # Tentukan path output (masuk ke folder results)
    csv_filename = f'hasil_baseline_random_{file_suffix}.csv'
    output_path = os.path.join('results', csv_filename)
    
    df_baseline.to_csv(output_path, index=False)
    print(f"[SUKSES] Data baseline disimpan ke: {output_path}")

    print("\n=== RANGKUMAN PERFORMA BASELINE (RANDOM) ===")
    print(f"Rata-rata Reward: {np.mean(history['rewards']):.2f}")
    print(f"Rata-rata Drops : {np.mean(history['drops']):.2f} per episode")
    print(f"Rata-rata Retries: {np.mean(history['retries']):.2f} per episode")
    print(f"Rata-rata QoS   : {np.mean(history['avg_qos']):.4f}")

if __name__ == "__main__":
    run_baseline()