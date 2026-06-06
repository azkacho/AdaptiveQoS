# config_rl.py

# General simulation settings
NUM_EPISODES = 700
TIMESTEPS_PER_EPISODE = 1000
PACKET_SIZE_BITS = 32 * 8  # 256 bits (32 Bytes)
RANDOM_SEED = 42

# --- 1. Parameter Jaringan WSN (Statik) ---
AREA_SIZE = 100
MAX_COMM_DISTANCE = 35
NUM_NODES = 50
SOURCE_NODES_COUNT = 20
NUM_PARENT_OPTIONS = 5
SINK_POSITION = (50, 50)

# --- 2. Definisi Aksi (Data Rate nRF24L01) ---
DATA_RATE_THRESHOLDS = {
    "low": 250_000,
    "medium": 1_000_000,
    "high": 2_000_000
}

# --- 3. Bobot untuk Fungsi Ganjaran / Biaya ---
# WEIGHTS = {
#     "packet_loss": 5.0,
#     "delay": 3.0,
#     "throughput": 2.0,
#     "rssi": 4.0,
#     "energy_penalty": 10.0,
#     "buffer_penalty": 8.0
# } fungsi lama, tidak dipakai lagi karena sudah ada QOS_WEIGHTS yang lebih spesifik dan terintegrasi dalam reward function.

# --- 4. Parameter Dinamis (Buffer & Energi) ---
MAX_BUFFER_CAPACITY = 100
INITIAL_ENERGY_JOULE = 1000
PACKET_ARRIVAL_RATE = 0.5

# --- 5. Parameter QoS & Reward ---
MAX_RETRANSMISSIONS = 10
PENALTY_DROP = 100.0
#ALPHA = 0.5 (perhitungan lama)
#BETA = 1.0 (perhitungan lama)

QOS_WEIGHTS = {
    "reliability": 0.7,
    "latency": 0.2,
    "energy": 0.1
}

MAX_EXPECTED_LATENCY = 0.1


# Karena durasi pengiriman pada 2 Mbps jauh lebih cepat, 
# konsumsi daya total dalam satuan joule untuk mengirim 
# sebuah paket data berukuran sama menjadi lebih kecil.
POWER_CONSUMPTION = {
    "low": 0.012,
    "medium": 0.011,
    "high": 0.010
}

# ============================================================
# --- 6. [BARU] Parameter EDF untuk Pelatihan (Opsi A) ---
# ============================================================

# Nilai EDF maksimum yang akan dilihat agen saat pelatihan.
# Diset sama dengan EDF paling ekstrem di Skenario B (badai = 0.9),
# sehingga distribusi training mencakup seluruh rentang pengujian.
EDF_TRAIN_MAX = 0.9

# Strategi sampling EDF saat pelatihan:
#   'random'     - EDF diambil acak seragam [0, EDF_TRAIN_MAX] tiap episode
#   'curriculum' - EDF dinaikkan bertahap (direkomendasikan)
EDF_TRAIN_STRATEGY = 'curriculum'

# Fase curriculum: (batas_episode, edf_max_fase)
# Fase 1: Ep 1-300   => EDF in [0.0, 0.30]  (kondisi ringan, bangun fondasi)
# Fase 2: Ep 301-600 => EDF in [0.0, 0.60]  (kondisi sedang)
# Fase 3: Ep 601-1000=> EDF in [0.0, 0.90]  (kondisi penuh, sesuai Skenario B)
EDF_CURRICULUM_PHASES = [
    (300,  0.30),
    (600,  0.60),
    (1000, 0.90),
]
