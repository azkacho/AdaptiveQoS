"""
Sub-analisis 5.1 — Konvergensi Training & Perbandingan Konfigurasi Model
========================================================================
Input  : Semua file train_log_wsn_dqn_E*_T*_*.csv di folder logs/
Output : analysis_output/01_training_convergence/
         - fig1_reward_convergence.png
         - fig2_qos_convergence.png
         - fig3_curriculum_edf.png
         - tabel1_training_summary.csv

Cara pakai:
    Jalankan dari root folder project:
    python analysis_training.py

    Jika logs/ ada di lokasi lain, ubah LOG_DIR di bawah.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# ── Konfigurasi Path ─────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
LOG_DIR    = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "analysis_output" / "training_convergence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Konfigurasi Figure (Format Akademis) ─────────────────────────────────────
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.titleweight': 'bold',
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linestyle':   '--',
    'lines.linewidth':  1.4,
})

# ── Konstanta Curriculum (dari config_rl.py) ─────────────────────────────────
EDF_CURRICULUM_PHASES = [(300, 0.30), (600, 0.60), (1000, 0.90)]
ROLLING_WINDOW        = 20   # window smoothing reward

# ── Palet Warna — otomatis assign per model, max 8 model ────────────────────
PALETTE = [
    '#1f77b4',  # biru
    '#d62728',  # merah
    '#2ca02c',  # hijau
    '#ff7f0e',  # oranye
    '#9467bd',  # ungu
    '#8c564b',  # coklat
    '#e377c2',  # pink
    '#17becf',  # cyan
]
LINESTYLES = ['-', '--', '-.', ':']


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PARSE SEMUA TRAINING LOGS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_model_label(filename: str) -> str:
    """
    Ekstrak label ringkas dari nama file.
    train_log_wsn_dqn_E200_T1000_20260508_150922.csv  →  E200 T1000
    """
    match = re.search(r'E(\d+)_T(\d+)', filename)
    if match:
        return f"E{match.group(1)} T{match.group(2)}"
    return Path(filename).stem


def load_all_training_logs(log_dir: Path) -> dict:
    """
    Membaca semua file train_log_wsn_dqn_*.csv dari log_dir.
    Mengembalikan dict: { label: DataFrame }
    Jika ada beberapa file dengan E dan T yang sama, ambil yang terbaru (mtime).
    """
    pattern = str(log_dir / "train_log_wsn_dqn_*.csv")
    all_files = glob.glob(pattern)

    if not all_files:
        raise FileNotFoundError(
            f"Tidak ditemukan file training log di: {log_dir}\n"
            f"Pastikan folder 'logs/' berisi file train_log_wsn_dqn_*.csv"
        )

    # Group by (E, T) — ambil file terbaru per konfigurasi
    groups = {}
    for fpath in all_files:
        fname = os.path.basename(fpath)
        match = re.search(r'E(\d+)_T(\d+)', fname)
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key not in groups or os.path.getmtime(fpath) > os.path.getmtime(groups[key]):
            groups[key] = fpath

    # Baca dan sortir berdasarkan (episodes, timesteps)
    models = {}
    for (ep, ts), fpath in sorted(groups.items()):
        label = f"E{ep} T{ts}"
        df = pd.read_csv(fpath)
        # Validasi kolom minimal
        required = {'Episode', 'Reward', 'Avg_QoS', 'EDF_Episode', 'Epsilon'}
        missing  = required - set(df.columns)
        if missing:
            print(f"  [SKIP] {label}: kolom hilang {missing}")
            continue
        models[label] = df
        print(f"  [OK] {label}: {len(df)} episodes, "
              f"EDF range [{df['EDF_Episode'].min():.2f}, {df['EDF_Episode'].max():.2f}], "
              f"epsilon akhir {df['Epsilon'].iloc[-1]:.4f}")

    return models


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FIGURE 1 — KONVERGENSI REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def plot_reward_convergence(models: dict, output_path: Path):
    """
    Line plot: Reward per episode (raw transparan + smoothed solid)
    untuk semua model dalam 1 axes.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for idx, (label, df) in enumerate(models.items()):
        color = PALETTE[idx % len(PALETTE)]
        ls    = LINESTYLES[idx % len(LINESTYLES)]
        ep    = df['Episode'].values
        rew   = df['Reward'].values

        # Raw — transparan, sangat tipis
        ax.plot(ep, rew, color=color, alpha=0.15, linewidth=0.7, linestyle=ls)

        # Smoothed — solid, tebal
        smoothed = pd.Series(rew).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        ax.plot(ep, smoothed.values, color=color, linewidth=1.8,
                linestyle=ls, label=label)

    # Garis horisontal reward=0 sebagai referensi
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':', alpha=0.5,
               label='Reward = 0 (baseline)')

    ax.set_title('Konvergensi Reward per Episode — Perbandingan Konfigurasi Model')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend(loc='lower right', framealpha=0.9)

    # Anotasi window smoothing
    ax.text(0.01, 0.02,
            f'Garis tebal = rolling mean (window={ROLLING_WINDOW})',
            transform=ax.transAxes, fontsize=8, color='#555555',
            verticalalignment='bottom')

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [SAVED] {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FIGURE 2 — KONVERGENSI QoS + EPSILON DECAY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_qos_epsilon(models: dict, output_path: Path):
    """
    2-panel figure:
    Panel atas  : Avg_QoS per episode (smoothed)
    Panel bawah : Epsilon decay — menunjukkan fase eksplorasi vs eksploitasi
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

    for idx, (label, df) in enumerate(models.items()):
        color = PALETTE[idx % len(PALETTE)]
        ls    = LINESTYLES[idx % len(LINESTYLES)]
        ep    = df['Episode'].values

        # Panel 1 — QoS
        qos      = df['Avg_QoS'].values
        qos_raw  = pd.Series(qos).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        ax1.plot(ep, qos, color=color, alpha=0.12, linewidth=0.7, linestyle=ls)
        ax1.plot(ep, qos_raw.values, color=color, linewidth=1.8,
                 linestyle=ls, label=label)

        # Panel 2 — Epsilon
        ax2.plot(ep, df['Epsilon'].values, color=color, linewidth=1.6,
                 linestyle=ls, label=label)

    # Panel 1 styling
    ax1.set_title('Konvergensi Rata-rata QoS (S_QoS) per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Avg QoS Score')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.9, color='#2ca02c', linewidth=0.8, linestyle=':',
                alpha=0.7, label='Target QoS = 0.9')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.text(0.01, 0.02,
             f'Garis tebal = rolling mean (window={ROLLING_WINDOW})',
             transform=ax1.transAxes, fontsize=8, color='#555555',
             verticalalignment='bottom')

    # Panel 2 styling
    ax2.set_title('Kurva Epsilon Decay (Eksplorasi → Eksploitasi)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon (ε)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.axhline(y=0.05, color='black', linewidth=0.8, linestyle=':',
                alpha=0.6, label='ε_min = 0.05')
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout(pad=3.0)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [SAVED] {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 3 — CURRICULUM EDF SCATTER
# ═══════════════════════════════════════════════════════════════════════════════

def plot_curriculum_edf(models: dict, output_path: Path):
    """
    Scatter EDF_Episode vs Reward per model.
    Satu subplot per model, dengan garis batas fase curriculum.
    Menunjukkan bagaimana curriculum learning mendistribusikan EDF.
    """
    n_models = len(models)
    ncols    = min(2, n_models)
    nrows    = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, (label, df) in enumerate(models.items()):
        ax    = axes_flat[idx]
        color = PALETTE[idx % len(PALETTE)]
        ep    = df['Episode'].values
        edf   = df['EDF_Episode'].values
        rew   = df['Reward'].values

        # Scatter: warna titik = reward (colormap)
        sc = ax.scatter(ep, edf, c=rew, cmap='RdYlGn',
                        s=6, alpha=0.6, linewidths=0,
                        vmin=min(rew), vmax=max(rew))

        # Garis batas fase curriculum (hanya untuk model yang melewatinya)
        phase_colors = ['#aaaaaa', '#888888', '#444444']
        max_ep = ep.max()
        for (ep_lim, edf_max), pc in zip(EDF_CURRICULUM_PHASES, phase_colors):
            if ep_lim <= max_ep:
                ax.axvline(x=ep_lim, color=pc, linewidth=0.9,
                           linestyle='--', alpha=0.7)
                ax.axhline(y=edf_max, color=pc, linewidth=0.7,
                           linestyle=':', alpha=0.5)
                ax.text(ep_lim + max_ep * 0.01, edf_max + 0.01,
                        f'EDF≤{edf_max}', fontsize=7, color=pc)

        # Colorbar per subplot
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('Reward', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_title(f'Distribusi EDF Training — {label}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('EDF Episode')
        ax.set_ylim(-0.05, 1.0)

    # Sembunyikan subplot kosong jika jumlah model ganjil
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        'Verifikasi Distribusi EDF per Fase Curriculum Learning\n'
        '(Warna titik = nilai reward; garis vertikal = batas fase)',
        fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout(pad=3.0)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [SAVED] {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TABEL RINGKASAN
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary_table(models: dict, output_path: Path) -> pd.DataFrame:
    """
    Tabel ringkasan: satu baris per model, berisi statistik akhir training.
    """
    rows = []
    for label, df in models.items():
        n        = len(df)
        pct10    = max(1, n // 10)
        last10   = df.tail(pct10)
        first10  = df.head(pct10)

        # Ekstrak E dan T dari label
        match = re.search(r'E(\d+)\s+T(\d+)', label)
        ep_cfg = int(match.group(1)) if match else 0
        ts_cfg = int(match.group(2)) if match else 0

        rows.append({
            'Model'                   : label,
            'Episodes'                : ep_cfg,
            'Timesteps/Ep'            : ts_cfg,
            'EDF Range'               : f"[{df['EDF_Episode'].min():.2f}, {df['EDF_Episode'].max():.2f}]",
            'Epsilon Akhir'           : round(df['Epsilon'].iloc[-1], 4),
            'Reward Mean (all)'       : round(df['Reward'].mean(), 2),
            'Reward Mean (last 10%)'  : round(last10['Reward'].mean(), 2),
            'Reward Std (last 10%)'   : round(last10['Reward'].std(), 2),
            'Reward Trend'            : f"{first10['Reward'].mean():.2f} → {last10['Reward'].mean():.2f}",
            'QoS Mean (last 10%)'     : round(last10['Avg_QoS'].mean(), 4),
            'QoS Std (last 10%)'      : round(last10['Avg_QoS'].std(), 4),
            'Total Drops'             : int(df['Drops'].sum()),
            'Avg Drops/Ep'            : round(df['Drops'].mean(), 2),
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_path, index=False)
    print(f"  [SAVED] {output_path.name}")

    # Tampilkan di konsol juga
    print()
    print(df_summary.to_string(index=False))
    return df_summary

# ── FUNGSI PENILAIAN MODEL TERBAIK ───────────────────────────────────────────
def evaluate_best_model(models_dict, output_dir):
    print("\n[5/5] Menganalisis dan Mencari Model Terbaik (Berdasarkan 50 Episode Terakhir)...")
    results = []
    
    for model_name, df in models_dict.items():
        # Kita ambil 50 episode terakhir karena di sinilah AI seharusnya sudah "Konvergen" (Pintar)
        last_50_episodes = df.tail(50)
        
        # Ujian Mental: Kita fokus pada performa AI saat lingkungan sedang BURUK (EDF >= 0.5)
        if 'EDF_Episode' in last_50_episodes.columns:
            hard_conditions = last_50_episodes[last_50_episodes['EDF_Episode'] >= 0.5]
            # Jika secara kebetulan 50 episode terakhir EDF-nya rendah semua, pakai semua data
            if hard_conditions.empty:
                hard_conditions = last_50_episodes
        else:
            hard_conditions = last_50_episodes
            
        # Hitung rata-rata ketahanan (QoS dan PSR) di fase akhir
        avg_psr = hard_conditions['PSR'].mean()
        avg_qos = hard_conditions['Avg_QoS'].mean()
        avg_reward = hard_conditions['Reward'].mean()
        
        # Rumus Skor: Menjumlahkan % PSR dan % QoS (Nilai maksimal adalah 200)
        score = (avg_psr * 100) + (avg_qos * 100)
        
        results.append({
            'Model_Name': model_name,
            'Total_Score': round(score, 2),
            'Avg_PSR_Akhir': round(avg_psr, 3),
            'Avg_QoS_Akhir': round(avg_qos, 3),
            'Avg_Reward_Akhir': round(avg_reward, 2)
        })
    
    # Jadikan DataFrame dan urutkan dari Skor Tertinggi ke Terendah
    results_df = pd.DataFrame(results).sort_values(by='Total_Score', ascending=False)
    
    # Print ke Terminal
    print("\n🏆 RANKING MODEL TERBAIK (Ujian Ketahanan EDF Tinggi):")
    print(results_df.to_string(index=False))
    
    # Simpan ke CSV untuk dimasukkan ke Bab 4 Skripsi
    out_path = output_dir / "tabel_ranking_model_terbaik.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n✅ Data ranking berhasil disimpan ke: {out_path}")
    
    return results_df
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Analisis — Konvergensi Training Model Adaptive QoS (DQN)")
    print("=" * 60)

    # Load semua training logs
    print(f"\n[1/5] Membaca training logs dari: {LOG_DIR}")
    models = load_all_training_logs(LOG_DIR)
    print(f"      {len(models)} model ditemukan: {list(models.keys())}")

    if not models:
        print("[ERROR] Tidak ada model yang berhasil dimuat. Periksa folder logs/")
        return

    # Figure 1 — Reward convergence
    print("\n[2/5] Membuat Figure 1: Konvergensi Reward...")
    plot_reward_convergence(
        models,
        OUTPUT_DIR / "fig1_reward_convergence.png"
    )

    # Figure 2 — QoS + Epsilon
    print("\n[3/5] Membuat Figure 2: Konvergensi QoS & Epsilon Decay...")
    plot_qos_epsilon(
        models,
        OUTPUT_DIR / "fig2_qos_epsilon.png"
    )

    # Figure 3 — Curriculum EDF scatter
    print("\n[4/5] Membuat Figure 3: Distribusi Curriculum EDF...")
    plot_curriculum_edf(
        models,
        OUTPUT_DIR / "fig3_curriculum_edf.png"
    )

    # Tabel ringkasan
    print("\n[5/5] Membuat Tabel Ringkasan...")
    build_summary_table(
        models,
        OUTPUT_DIR / "tabel1_training_summary.csv"
    )

    evaluate_best_model(models, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"[SELESAI] Semua output tersimpan di:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
