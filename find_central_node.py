# find_central_node.py
# Jalankan SATU KALI sebelum semua run Skenario C.
# Output: TARGET_NODE yang digunakan tetap di semua 16 run.

import sys, os, random, collections, glob
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
from wsn_rl_env import WSN_RL_Env
from config_rl import RANDOM_SEED, SOURCE_NODES_COUNT

# ── Arsitektur DQN (identik dengan main-rl.py) ────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ── Konfigurasi ────────────────────────────────────────────────────────────
SIMULATION_STEPS = 300   # langkah simulasi per model/mode
SEED             = RANDOM_SEED

# ── Load semua model .pth yang tersedia ───────────────────────────────────
def load_models(env):
    pth_files = sorted(glob.glob("AdaptiveQoS/models/wsn_dqn_E500_T1000*.pth"))
    if not pth_files:
        print("[ERROR] Tidak ada file .pth di folder models/")
        sys.exit(1)

    models = {}
    for pth in pth_files:
        name = os.path.basename(pth).replace("wsn_dqn_","").replace(".pth","")
        # Ambil hanya bagian E dan T: misal E1000_T500
        parts = name.split("_")
        label = f"{parts[0]}_{parts[1]}"
        m = DQN(env.observation_space.shape[0], env.action_space.n)
        m.load_state_dict(torch.load(pth, map_location='cpu'))
        m.eval()
        models[label] = m
        print(f"  [LOAD] {label} dari {os.path.basename(pth)}")
    return models

# ── Simulasi routing dan hitung frekuensi parent ──────────────────────────
def count_parent_usage(env, model_or_none, steps, label):
    """
    Simulasi SIMULATION_STEPS langkah.
    Setiap langkah: pilih source node → ambil keputusan routing →
    catat siapa yang dipilih sebagai parent relay (bukan sink).
    """
    usage  = collections.Counter()
    source_nodes = list(range(SOURCE_NODES_COUNT))  # node 0-19 sebagai source

    for step in range(steps):
        for node_id in source_nodes:
            env.current_node_id = node_id
            node_state, _ = env._get_state_for_node(node_id)

            if model_or_none is not None:
                # Mode AI
                with torch.no_grad():
                    st = torch.FloatTensor(node_state).unsqueeze(0)
                    action = model_or_none(st).argmax().item()
            else:
                # Mode Greedy
                action = env.get_greedy_action()

            parent_id, _, _ = env._decode_action(action)

            # Hanya catat jika parent adalah node biasa, bukan sink langsung
            if parent_id != 'sink':
                usage[parent_id] += 1

    print(f"  [{label:18s}] Top 3 relay: " +
          ", ".join([f"Node {n}({c}x)" for n, c in usage.most_common(3)]))
    return usage

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  IDENTIFIKASI TARGET NODE — Skenario C")
    print("=" * 60)

    # Inisialisasi environment dengan seed tetap
    env = WSN_RL_Env()
    env.reset(seed=SEED)
    env.edf = 0.0  # kondisi normal, tanpa EDF

    print(f"\n[INFO] Topologi: seed={SEED}, {env.num_nodes} node")
    print(f"[INFO] Menjalankan {SIMULATION_STEPS} step per mode/model\n")

    print("[STEP 1] Memuat model DQN...")
    models = load_models(env)

    # Akumulasi usage dari SEMUA model dan mode
    aggregate = collections.Counter()

    print("\n[STEP 2] Menghitung frekuensi node relay per mode/model...")

    # Semua model DQN
    for label, model in models.items():
        usage = count_parent_usage(env, model, SIMULATION_STEPS, f"AI {label}")
        aggregate.update(usage)

    # Greedy
    usage_greedy = count_parent_usage(env, None, SIMULATION_STEPS, "GREEDY")
    aggregate.update(usage_greedy)

    # ── Hasil ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HASIL — Node Relay Terbanyak Digunakan (Semua Mode + Model)")
    print("=" * 60)
    print(f"\n{'Rank':<6} {'Node ID':<10} {'Total Dipilih':<16} {'Posisi (x,y)':<20} {'Jarak ke SINK'}")
    print("-" * 65)

    sink_x, sink_y = env.positions['sink']
    for rank, (node_id, count) in enumerate(aggregate.most_common(10), 1):
        px, py = env.positions[node_id]
        dist_to_sink = ((px-sink_x)**2 + (py-sink_y)**2)**0.5
        marker = "  ← REKOMENDASI TARGET" if rank == 1 else ""
        print(f"  {rank:<4} Node {node_id:<5} {count:<16} ({px:5.1f}, {py:5.1f})     {dist_to_sink:5.1f} m{marker}")

    TARGET_NODE = aggregate.most_common(1)[0][0]
    px, py = env.positions[TARGET_NODE]

    print("\n" + "=" * 60)
    print(f"  TARGET_NODE  = {TARGET_NODE}")
    print(f"  Posisi       = ({px:.1f}, {py:.1f})")
    print(f"  Jarak ke SINK= {((px-sink_x)**2+(py-sink_y)**2)**0.5:.1f} m")
    print("=" * 60)
    print("\n[PENTING] Catat angka TARGET_NODE di atas.")
    print("          Gunakan node ini untuk run Skenario C.")

if __name__ == "__main__":
    main()