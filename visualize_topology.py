import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import os
import glob
from wsn_rl_env import WSN_RL_Env
from config_rl import SINK_POSITION

# --- 1. Struktur Model (Wajib Sama dengan main-rl.py) ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- 2. Fungsi Tracing Jalur (Dinamis & Akurat) ---
def trace_route(env, model, start_node, num_nodes):
    path_edges = []
    current = start_node
    visited = {current}
    
    for _ in range(num_nodes):
        # Mengatur ID node agar state & decode_action sinkron dengan posisi saat ini
        env.current_node_id = current
        try:
            # env._get_state_for_node sekarang membaca state dari current_node_id yang diset di atas
            node_state, _ = env._get_state_for_node(current)
        except: break

        with torch.no_grad():
            state_tensor = torch.FloatTensor(node_state).unsqueeze(0)
            action = model(state_tensor).argmax().item()
        
        try:
            # Mengambil keputusan parent berdasarkan mapping lingkungan saat ini
            parent_id, _, _ = env._decode_action(action)
        except: break
            
        actual_parent = 'sink' if parent_id == num_nodes else parent_id
        path_edges.append((current, actual_parent))
        
        if actual_parent == 'sink' or actual_parent in visited: break
        current = actual_parent
        visited.add(current)
    return path_edges

# --- 3. Fungsi Utama Visualisasi (Dark Mode) ---
def visualize_dynamic_dark():
    print("[INFO] Menghasilkan Visualisasi Topologi Modern...")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    RESULT_DIR = os.path.join(BASE_DIR, "results")

    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)

    # 1. Cari file model pth terbaru menggunakan glob
    search_pattern = os.path.join(MODEL_DIR, "wsn_dqn_E500_T1000*.pth")
    #--- UBAH NAMA PTH SESUAI NAMA MODEL---#
    list_of_models = glob.glob(search_pattern)

    if not list_of_models:
        print("[ERROR] Tidak ditemukan file model wsn_dqn_E*.pth di folder models.")
        return

    latest_model = max(list_of_models, key=os.path.getmtime)
    print(f"[INFO] Membaca model dari: {os.path.basename(latest_model)}")

    # 2. Ekstrak parameter menggunakan metode split
    # Asumsi nama file: wsn_dqn_E1000_T200_20260413_095650.pth
    filename_only = os.path.basename(latest_model).replace(".pth", "")
    parts = filename_only.split("_")
    
    e_val = parts[-4]
    t_val = parts[-3]
    timestamp = f"{parts[-2]}_{parts[-1]}"
    
    # Menentukan nama file output secara dinamis
    output_filename = f"DQN_topology_{e_val}_{t_val}_{timestamp}.png"
    OUTPUT_IMG = os.path.join(RESULT_DIR, output_filename)

    # Inisialisasi Env TANPA SEED (Layout akan berubah tiap kali run)
    env = WSN_RL_Env()
    env.reset()
    
    # Load Model .pth
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(latest_model))
    model.eval()

    G = nx.DiGraph()
    # Menggunakan atribut 'positions' yang terdeteksi di env Anda
    pos = {i: env.positions[i] for i in range(env.num_nodes)}
    pos['sink'] = np.array(SINK_POSITION)

    # --- KONFIGURASI GAYA DARK MODE ---
    color_bg = '#313131'        # Latar belakang figur hitam
    color_node_bg = 'grey'    # Node abu-abu terlihat jelas
    color_sink = '#E74C3C'    # Kotak Sink (Merah Modern)
    color_text = 'white'      # Teks judul putih
    color_grid = '#555555'    # Grid grey samar (Tipis)
    
    # Palette Warna Jalur Modern (Material Design) yang Pop di Hitam
    path_colors = ['#3498DB', '#2ECC71', '#E67E22'] # Blue, Green, Orange
    
    # Membuat figur dengan facecolor hitam
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=color_bg)
    ax.set_facecolor(color_bg)

    # 1. Gambar Semua Node Dasar (Abu-abu, Linewidth Tipis)
    nx.draw_networkx_nodes(G, pos, nodelist=range(env.num_nodes), 
                           node_color=color_node_bg, node_size=60, 
                           edgecolors='#A0A0A0', linewidths=0.5, ax=ax)
    
    # 2. Gambar Sink (Kotak Modern Red)
    nx.draw_networkx_nodes(G, pos, nodelist=['sink'], 
                           node_color=color_sink, node_size=300, 
                           node_shape='s', edgecolors=color_sink, ax=ax)

    # 3. Pilih 3 Source Node Secara Acak (Dinamis)
    all_nodes = list(range(env.num_nodes))
    source_nodes = np.random.choice(all_nodes, 3, replace=False)

    # 4. Tracing & Gambar Jalur dari Setiap Source
    for i, src in enumerate(source_nodes):
        edges = trace_route(env, model, src, env.num_nodes)
        
        if not edges: continue
        
        color = path_colors[i]
        
        # Gambar Jalur dengan Gaya Modern (Garis Melengkung)
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, 
                               width=2.0, arrows=True, arrowsize=12, 
                               connectionstyle="arc3,rad=0.1", ax=ax)
        
        # Tandai Source Node dengan Warna Path
        nx.draw_networkx_nodes(G, pos, nodelist=[src], 
                               node_color=color, node_size=180, ax=ax)

    # 5. Sentuhan Akhir Minimalis & Kekinian
    ax.set_title("WSN Routing path based on Trained DQN Agent Decisions", 
                 fontsize=14, fontweight='bold', color=color_text, pad=20)
    
    # Subtle Grid Grey Samaritan
    ax.grid(True, linestyle='-', color=color_grid, linewidth=0.5, alpha=0.6)
    
    # Menghapus 'Spines' (Kotak axes) agar minimalis
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Tick Marks Minimalis & Teks Light-grey
    ax.tick_params(colors='#BBBBBB', labelsize=9)
    ax.set_xlabel("X Position (meter)", color='#BBBBBB', fontsize=10)
    ax.set_ylabel("Y Position (meter)", color='#BBBBBB', fontsize=10)

    # Simpan ke folder results
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"[SUKSES] Grafik Dark Mode disimpan di: {OUTPUT_IMG}")
    plt.show() # Tampilkan jika ingin melihat langsung

if __name__ == "__main__":
    visualize_dynamic_dark()