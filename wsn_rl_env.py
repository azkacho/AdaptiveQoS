import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
import math

from config_rl import (
    NUM_NODES, AREA_SIZE, MAX_COMM_DISTANCE, SINK_POSITION,
    DATA_RATE_THRESHOLDS, WEIGHTS, MAX_BUFFER_CAPACITY,
    POWER_CONSUMPTION, NUM_PARENT_OPTIONS,
    PACKET_SIZE_BITS, PACKET_ARRIVAL_RATE, TIMESTEPS_PER_EPISODE,
    MAX_RETRANSMISSIONS, PENALTY_DROP, ALPHA, BETA,
    QOS_WEIGHTS, MAX_EXPECTED_LATENCY, INITIAL_ENERGY_JOULE, RANDOM_SEED,
    EDF_TRAIN_MAX, EDF_TRAIN_STRATEGY, EDF_CURRICULUM_PHASES
)

# ── Konstanta State Space ───────────────────────────────────────────────────
NUM_LINK_METRICS    = 4   # p_loss, latency, score_rel, score_eng
NUM_DYNAMIC_METRICS = 2   # buffer_norm, energy_norm
# [BARU] +1 untuk EDF — agen perlu "melihat" kondisi lingkungan agar bisa
# membuat keputusan PROAKTIF, bukan hanya reaktif via p_loss tinggi.
NUM_EDF_METRICS     = 1
STATE_DIMENSION     = NUM_LINK_METRICS + NUM_DYNAMIC_METRICS + NUM_EDF_METRICS  # = 7

# ── Konstanta Action Space ──────────────────────────────────────────────────
DATA_RATE_OPTIONS = len(DATA_RATE_THRESHOLDS)
NUM_ACTIONS       = NUM_PARENT_OPTIONS * DATA_RATE_OPTIONS


class WSN_RL_Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(WSN_RL_Env, self).__init__()

        self.max_comm_distance = MAX_COMM_DISTANCE

        self.num_nodes    = NUM_NODES
        self.positions    = {}
        self.node_states  = {}
        self.network_graph     = None
        self.current_node_id   = None
        self.parent_mapping    = {}
        self.last_link_qos_metrics = np.zeros(NUM_LINK_METRICS)
        self.current_step_count    = 0

        # [BARU] Atribut tunggal untuk EDF.
        # - Saat training (step): diisi oleh reset() sesuai strategi.
        # - Saat Dashboard (step_for_node): ditimpa oleh logic.py via env.edf = ...
        self.edf = 0.0

        # [BARU] Penghitung episode internal (digunakan curriculum)
        self._episode_count = 0

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIMENSION,), dtype=np.float32
        )

        self._initialize_network_graph()
        self.reset()

    # ── Inisialisasi Jaringan ────────────────────────────────────────────────

    def _initialize_network_graph(self):
        if not self.positions:
            random.seed(RANDOM_SEED)       
            np.random.seed(RANDOM_SEED) 
            self.positions = {
                i: (random.uniform(0, AREA_SIZE), random.uniform(0, AREA_SIZE))
                for i in range(self.num_nodes)
            }
            self.positions['sink'] = SINK_POSITION
        else:
            self.positions['sink'] = SINK_POSITION

        self.network_graph = nx.Graph()
        self.network_graph.add_nodes_from(range(self.num_nodes))
        self.network_graph.add_node('sink')
        self._precalculate_links_and_parents()

    def _precalculate_links_and_parents(self):
        for i in range(self.num_nodes):
            neighbors = [
                j for j in self.network_graph.nodes
                if j != i and self._get_distance(i, j) <= MAX_COMM_DISTANCE
            ]
            if not neighbors:
                distances = {j: self._get_distance(i, j) for j in self.network_graph.nodes if j != i}
                neighbors = [min(distances, key=distances.get)]
            self.parent_mapping[i] = random.sample(neighbors, min(NUM_PARENT_OPTIONS, len(neighbors)))

    def _get_distance(self, u, v):
        if u not in self.positions or v not in self.positions:
            return MAX_COMM_DISTANCE + 100
        xu, yu = self.positions[u]
        xv, yv = self.positions[v]
        return np.hypot(xu - xv, yu - yv)

    def _decode_action(self, action):
        parent_idx = action // DATA_RATE_OPTIONS
        dr_idx     = action % DATA_RATE_OPTIONS
        available_parents = self.parent_mapping.get(self.current_node_id, [])
        parent_id = 'sink' if not available_parents \
                    else available_parents[parent_idx % len(available_parents)]
        data_rate_keys = list(DATA_RATE_THRESHOLDS.keys())
        data_rate_key  = data_rate_keys[dr_idx]
        data_rate_bps  = DATA_RATE_THRESHOLDS[data_rate_key]
        return parent_id, data_rate_bps, data_rate_key

    # ── Model Fisika Kanal ───────────────────────────────────────────────────

    def calculate_link_quality(self, distance, data_rate, edf=0.0):
        """
        Model hybrid: Fisika nRF24L01 + Environmental Degradation Factor (EDF).
        Output: Probabilitas Packet Loss — 0.0 (sempurna) hingga 1.0 (total gagal).

        Rumus:
            P_loss_final = 1 - [ (1 - P_loss_fisik) * (1 - EDF) ]

        Intuisi: EDF menekan "kualitas sinyal bersih" secara multiplicative.
        Pada EDF=0.9, bahkan link fisika bagus (P_fis=0.05) menghasilkan
        P_final = 1 - (0.95 * 0.1) = 0.905 — mendekati kegagalan total.
        """
        sensitivity_limit = {250_000: 45.0, 1_000_000: 30.0, 2_000_000: 15.0}
        limit    = sensitivity_limit.get(data_rate, 20.0)
        phys_loss = (0.1 + (distance - limit) * 0.1) if distance > limit \
                    else (0.01 + (distance / limit) * 0.05)
        phys_loss = min(1.0, max(0.0, phys_loss))

        degraded_quality = (1.0 - phys_loss) * (1.0 - edf)
        return min(1.0, max(0.0, 1.0 - degraded_quality))

    # ── [BARU] Sampling EDF untuk Training ──────────────────────────────────

    def _sample_training_edf(self) -> float:
        """
        Menentukan nilai EDF untuk satu episode pelatihan.

        Strategi 'curriculum' (default, direkomendasikan):
            Membagi 1000 episode menjadi 3 fase dengan rentang EDF yang
            semakin lebar. Agen terlebih dahulu menguasai lingkungan bersih,
            kemudian secara bertahap diperkenalkan kondisi degradasi yang
            lebih parah. Ini mencegah divergensi awal yang dapat terjadi
            jika agen langsung menghadapi EDF ekstrem.

        Strategi 'random':
            EDF diambil acak seragam dari [0, EDF_TRAIN_MAX] setiap episode.
            Lebih sederhana tetapi konvergensi bisa lebih lambat karena
            agen melihat kondisi sulit sejak episode pertama.
        """
        if EDF_TRAIN_STRATEGY == 'curriculum':
            current_ep  = self._episode_count
            edf_max_now = EDF_TRAIN_MAX  # fallback
            for phase_limit, phase_edf_max in EDF_CURRICULUM_PHASES:
                if current_ep <= phase_limit:
                    edf_max_now = phase_edf_max
                    break
            return float(np.random.uniform(0.0, edf_max_now))
        else:  # 'random'
            return float(np.random.uniform(0.0, EDF_TRAIN_MAX))

    # ── Greedy Baseline ──────────────────────────────────────────────────────

    def get_greedy_action(self):
        """
        Algoritma Greedy: pilih parent terdekat ke sink, data rate medium.
        Mengembalikan action_index agar kompatibel dengan step().
        """
        node_id   = self.current_node_id
        neighbors = self.parent_mapping.get(node_id, [])
        if not neighbors:
            return 0

        sink_x, sink_y = self.positions['sink']
        best_idx, min_dist = 0, float('inf')
        for idx, nid in enumerate(neighbors):
            nx_, ny_ = self.positions[nid]
            d = math.sqrt((nx_ - sink_x)**2 + (ny_ - sink_y)**2)
            if d < min_dist:
                min_dist, best_idx = d, idx

        num_rates      = len(DATA_RATE_THRESHOLDS)
        medium_rate_idx = 1 if num_rates > 1 else 0
        return (best_idx * num_rates) + medium_rate_idx

    # ── Step (digunakan saat Training) ──────────────────────────────────────

    def step(self, action):
        """
        Eksekusi satu langkah simulasi saat PELATIHAN.

        Perubahan Opsi A:
            EDF kini diambil dari self.edf (diset oleh reset() sesuai strategi
            curriculum/random), bukan lagi hardcoded edf=0.0.
            Ini memastikan agen dilatih menghadapi berbagai kondisi lingkungan.
        """
        self.current_step_count += 1
        parent_id, data_rate_bps, dr_key = self._decode_action(action)
        dist   = self._get_distance(self.current_node_id, parent_id)

        # [PERUBAHAN KUNCI] Gunakan self.edf — bukan lagi 0.0
        p_loss = self.calculate_link_quality(dist, data_rate_bps, edf=self.edf)

        num_retries = 0
        is_dropped  = False
        while random.random() < p_loss:
            num_retries += 1
            if num_retries >= MAX_RETRANSMISSIONS:
                is_dropped = True
                break

        tx_time      = PACKET_SIZE_BITS / data_rate_bps
        actual_latency = tx_time + (num_retries * 0.001)
        power_w      = POWER_CONSUMPTION[dr_key]
        energy_spent = power_w * actual_latency

        self.node_states[self.current_node_id]['energy'] -= energy_spent
        current_energy = self.node_states[self.current_node_id]['energy']

        score_rel = 0.0 if is_dropped else 1.0
        score_lat = max(0, 1 - (actual_latency / MAX_EXPECTED_LATENCY))
        score_eng = max(0, current_energy / INITIAL_ENERGY_JOULE)

        S_QoS = (QOS_WEIGHTS['reliability'] * score_rel +
                 QOS_WEIGHTS['latency']     * score_lat +
                 QOS_WEIGHTS['energy']      * score_eng)

        self.last_link_qos_metrics = np.array(
            [p_loss, actual_latency, score_rel, score_eng], dtype=np.float32
        )

        if is_dropped:
            # ── Drop: sinyal negatif terkuat, tapi BUKAN terminal state ─────────
            # terminated = False → episode lanjut ke t+1
            # Efek Bellman: Q(s,a) ← -1.0 + γ·max Q(s',a')
            #   → agen tahu ada masa depan setelah drop, dan belajar hindari drop
            #   berikutnya (bukan hanya menghindari state saat drop terjadi).
            # Nilai -1.0 selalu lebih buruk dari kondisi retry terparah:
            #   retry_ratio → 1 ⟹ reward → -MAX_RETRANS/(MAX_RETRANS) = -1.0
            #   tapi drop terjadi SETELAH batas retry, jadi hard-coded -1.0 < semua retry
            reward     = -1.0
            terminated = False

        else:
            # ── Non-drop: reward self-normalized dalam range (-1, +S_QoS] ───────
            # Formula: reward = S_QoS*(1 - retry_ratio) - retry_ratio
            #
            # Properti kunci:
            #   • 0 retries → reward = S_QoS ∈ [0, 1]     (terbaik)
            #   • MAX-1 retries, QoS=0.5 → reward ≈ -0.70  (buruk tapi > -1.0)
            #   • Drop (di atas) → reward = -1.0            (terburuk)
            # Urutan terjamin: drop < max_retry < ... < 0_retry
            # Tidak bergantung pada ALPHA/BETA — stabil di berbagai konfigurasi
            retry_ratio = num_retries / MAX_RETRANSMISSIONS   # ∈ [0, 1)
            reward      = S_QoS * (1.0 - retry_ratio) - retry_ratio
            terminated  = False

        if current_energy <= 0:
            # ── Energi habis: satu-satunya true terminal state ───────────────────
            terminated = True
            reward    -= 1.0   # penalti tambahan, proporsional skala baru

        truncated = self.current_step_count >= TIMESTEPS_PER_EPISODE
        self.current_node_id = random.choice(list(self.node_states.keys()))
        next_state, _ = self._get_state_for_node(self.current_node_id)

        info = {
            "retries": num_retries,
            "dropped": is_dropped,
            "S_QoS":   S_QoS,
            "distance": dist,
            "rate":    dr_key,
            "edf":     self.edf,      # [BARU] log EDF aktual episode ini
        }
        return next_state, reward, terminated, truncated, info

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        self._episode_count    += 1
        self._initialize_network_graph()
        self.node_states = {
            i: {'buffer': 0, 'energy': INITIAL_ENERGY_JOULE}
            for i in range(self.num_nodes)
        }
        self.current_node_id = random.choice(list(self.node_states.keys()))
        self.last_link_qos_metrics = np.zeros(NUM_LINK_METRICS)

        # [BARU] Sampling EDF baru untuk episode ini
        self.edf = self._sample_training_edf()

        initial_state, _ = self._get_state_for_node(self.current_node_id, initial=True)
        return initial_state, {'positions': self.positions}

    # ── Kalkulasi State ──────────────────────────────────────────────────────

    def _get_state_for_node(self, node_id, initial=False):
        """
        Menghasilkan vektor state untuk satu node.

        [PERUBAHAN OPSI A]
        State lama (dim=6): [p_loss, latency, score_rel, score_eng, buffer, energy]
        State baru (dim=7): [p_loss, latency, score_rel, score_eng, buffer, energy, EDF]

        Menambahkan EDF ke state membuat agen mampu membedakan:
          - "p_loss tinggi karena jarak jauh" (fisika buruk, pilih low rate)
          - "p_loss tinggi karena EDF = 0.9"  (cuaca buruk, pilih low rate + hop pendek)
        Tanpa EDF di state, agen hanya bisa bereaksi terhadap konsekuensi (p_loss tinggi)
        tetapi tidak memahami KONTEKS penyebabnya, sehingga tidak bisa proaktif.
        """
        if initial:
            link_metrics = np.array([0.0, 0.0, 1.0, 1.0])
        else:
            link_metrics = self.last_link_qos_metrics

        buffer_norm = self.node_states[node_id]['buffer'] / MAX_BUFFER_CAPACITY
        energy_norm = self.node_states[node_id]['energy'] / INITIAL_ENERGY_JOULE

        # [BARU] EDF sudah dinormalisasi [0,1] — tidak perlu transformasi tambahan
        edf_norm = float(np.clip(self.edf, 0.0, 1.0))

        state = np.concatenate(
            [link_metrics, [buffer_norm, energy_norm, edf_norm]]
        ).astype(np.float32)

        terminated = self.node_states[node_id]['energy'] <= 0
        return state, terminated

    # ── Step for Dashboard (Digital Twin) ───────────────────────────────────

    def step_for_node(self, node_id, action, node_states_dict):
        """
        Simulasi transmisi per node untuk Digital Twin.
        EDF diambil dari self.edf yang ditimpa logic.py via: env.edf = current_edf
        Tidak ada perubahan di fungsi ini — kompatibel dengan logic.py yang ada.
        """
        parent_id, data_rate_bps, dr_key = self._decode_action(action)
        dist = self._get_distance(int(node_id), parent_id)

        # self.edf sudah diset oleh logic.py sebelum memanggil fungsi ini
        p_loss = self.calculate_link_quality(dist, data_rate_bps, edf=self.edf)

        num_retries = 0
        is_dropped  = False
        while random.random() < p_loss:
            num_retries += 1
            if num_retries >= MAX_RETRANSMISSIONS:
                is_dropped = True
                break

        info = {
            "retries":  num_retries,
            "dropped":  is_dropped,
            "distance": dist,
            "rate":     dr_key
        }
        return None, 0, False, False, info