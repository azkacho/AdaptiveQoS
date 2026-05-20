import torch
import torch.nn as nn
import numpy as np
import math
import random
import sys
import os

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from wsn_rl_env import WSN_RL_Env
from config_rl import (
    NUM_NODES, INITIAL_ENERGY_JOULE, MAX_COMM_DISTANCE,
    DATA_RATE_THRESHOLDS, SINK_POSITION, RANDOM_SEED
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

SCALE = 8

# ── DQN Architecture ────────────────────────────────────────────────────
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

# ── Singleton Environment ────────────────────────────────────────────────
env = WSN_RL_Env()
initial_positions_raw = env.positions
state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

# ── Active model state ───────────────────────────────────────────────────
model = None
_active_model_info = {
    'name':   'No Model Loaded',
    'path':   None,
    'loaded': False,
}

# ── Internal loader ──────────────────────────────────────────────────────
def _try_load(path: str) -> bool:
    global model
    try:
        candidate = DQN(state_dim, action_dim)
        candidate.load_state_dict(
            torch.load(path, map_location=torch.device('cpu'))
        )
        candidate.eval()
        model = candidate
        return True
    except Exception as e:
        print(f"[ERROR] Gagal memuat model dari {path}: {e}")
        return False

# ── Startup: load default model or fallback ──────────────────────────────
_models_dir   = os.path.join(BASE_DIR, "models")
_default_path = os.path.join(_models_dir, "wsn_dqn_model.pth") #----------------------------------------------------------------------

if os.path.isfile(_default_path) and _try_load(_default_path):
    _active_model_info.update({'name': os.path.basename(_default_path),
                                'path': _default_path, 'loaded': True})
    print(f"[INFO] Model default dimuat: {_default_path}")
elif os.path.isdir(_models_dir):
    for _f in sorted(os.listdir(_models_dir), reverse=True):
        if _f.endswith('.pth'):
            _p = os.path.join(_models_dir, _f)
            if _try_load(_p):
                _active_model_info.update({'name': _f, 'path': _p, 'loaded': True})
                print(f"[INFO] Model fallback dimuat: {_p}")
                break

# ── Public API ───────────────────────────────────────────────────────────
def load_model_from_path(new_path: str):
    """Return (success: bool, message: str)"""
    global model
    if not os.path.isfile(new_path):
        return False, f"File tidak ditemukan: {new_path}"
    if _try_load(new_path):
        _active_model_info.update({
            'name':   os.path.basename(new_path),
            'path':   new_path,
            'loaded': True,
        })
        return True, os.path.basename(new_path)
    return False, f"Gagal memuat: {os.path.basename(new_path)}"


def get_active_model_info() -> dict:
    return _active_model_info.copy()


def get_available_models(models_dir: str = None) -> list:
    if models_dir is None:
        models_dir = _models_dir
    if not os.path.isdir(models_dir):
        return []
    result = []
    for fname in sorted(os.listdir(models_dir), reverse=True):
        if not fname.lower().endswith('.pth'):
            continue
        fpath = os.path.join(models_dir, fname)
        result.append({
            'filename':  fname,
            'path':      fpath,
            'size_kb':   round(os.path.getsize(fpath) / 1024, 1),
            'is_active': (fpath == _active_model_info.get('path')),
        })
    return result


# ── Main simulation logic ────────────────────────────────────────────────
def get_logic_state(current_positions, node_data, old_routing_data,
                    override_range, edf_val=0.0, mode_val='AI', sel_id=None):
    elements         = []
    total_energy     = 0
    total_signal     = 0
    active_nodes     = 0
    new_routing_data = {}
    events           = []

    visual_range = override_range if override_range else MAX_COMM_DISTANCE
    current_edf  = edf_val if edf_val is not None else 0.0

    env.positions = {}
    for k, v in current_positions.items():
        if k == 'sink':
            env.positions['sink'] = (v[0], v[1])
        else:
            env.positions[int(k)] = (v[0], v[1])

    for i in range(NUM_NODES):
        s_i  = str(i)
        data = node_data.get(s_i, {'energy': INITIAL_ENERGY_JOULE})
        eng  = data['energy']
        total_energy += eng
        if eng > 0:
            active_nodes += 1

        ratio    = eng / INITIAL_ENERGY_JOULE
        bg_color = '#3498db' if mode_val == 'AI' else '#9b59b6'
        classes  = 'sensor-node'

        if ratio < 0.2:
            bg_color = '#e74c3c'
        if eng <= 0:
            bg_color = '#2c3e50'
            classes += ' dead-node'
            was_alive    = old_routing_data.get(s_i, {}).get('alive', True)
            already_dead = old_routing_data.get(s_i, {}).get('dead_logged', False)
            if was_alive and not already_dead:
                events.append(f"💀 Node {s_i} DIED (Battery Depleted)")

        if i == env.current_node_id:
            classes += ' active-node'
            bg_color = '#2ecc71'

        # Tandai node yang dipilih user (inspector selection)
        if sel_id is not None and str(i) == str(sel_id):
            classes += ' selected-node'

        raw_x = current_positions.get(s_i, [0, 0])[0]
        raw_y = current_positions.get(s_i, [0, 0])[1]
        elements.append({
            'data':     {'id': s_i, 'label': s_i, 'energy': eng},
            'position': {'x': raw_x * SCALE, 'y': raw_y * SCALE},
            'classes':  classes,
            'style':    {'background-color': bg_color}
        })

    sink_pos = current_positions.get('sink', [50, 50])
    elements.append({
        'data':     {'id': 'sink', 'label': 'SINK', 'type': 'sink'},
        'position': {'x': sink_pos[0] * SCALE, 'y': sink_pos[1] * SCALE},
        'classes':  'sink-node'
    })

    def calc_dist_px(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    all_keys = list(current_positions.keys())
    for i in range(len(all_keys)):
        for j in range(i + 1, len(all_keys)):
            id1, id2 = all_keys[i], all_keys[j]
            p1 = (current_positions[id1][0]*SCALE, current_positions[id1][1]*SCALE)
            p2 = (current_positions[id2][0]*SCALE, current_positions[id2][1]*SCALE)
            if calc_dist_px(p1, p2) <= (visual_range * SCALE):
                elements.append({
                    'data': {'source': str(id1), 'target': str(id2), 'type': 'ghost'},
                    'classes': 'ghost-edge'
                })

    for i in range(NUM_NODES):
        s_i = str(i)
        if node_data.get(s_i, {'energy': 0})['energy'] <= 0:
            new_routing_data[s_i] = {
                'parent':      None,
                'alive':       False,
                'dropped':     True,
                'retries':     0,
                'dead_logged': True,   # cegah duplikasi event DIED
            }
            continue

        env.current_node_id = i
        env.edf = current_edf
        state, _ = env._get_state_for_node(i)

        if mode_val == 'AI' and model is not None:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = model(state_t).argmax().item()
        else:
            action = env.get_greedy_action()

        parent_id, data_rate_bps, dr_key = env._decode_action(action)
        _, _, _, _, info = env.step_for_node(s_i, action, node_data)

        is_alive = True
        if str(parent_id) != 'sink' and parent_id is not None:
            if node_data.get(str(parent_id), {'energy': 0})['energy'] <= 0:
                is_alive  = False
                parent_id = None

        if parent_id is not None and is_alive:
            dist_real = env._get_distance(i, parent_id)
            base_q    = max(0, 1 - (dist_real / visual_range))
            final_q   = base_q * (1.0 - current_edf)
            total_signal += final_q

            color = '#e67e22'
            if final_q > 0.7:   color = '#2ecc71'
            elif final_q < 0.3: color = '#e74c3c'

            width = 2
            if "high" in dr_key:  width = 4
            elif "low" in dr_key: width = 1

            elements.append({
                'data': {
                    'source': s_i, 'target': str(parent_id),
                    'type': 'active', 'color': color, 'width': width
                },
                'classes': 'active-edge'
            })
            new_routing_data[s_i] = {
                'parent': parent_id, 'dr': dr_key,
                'dist':   dist_real, 'alive': True
            }
            if s_i in old_routing_data:
                prev_p = old_routing_data[s_i].get('parent')
                if str(prev_p) != str(parent_id) and prev_p is not None:
                    events.append(f" Node {s_i}: Route Switch {prev_p} ➝ {parent_id}")
            else:
                events.append(f" Node {s_i} connected to {parent_id}")
        else:
            if s_i in old_routing_data and old_routing_data[s_i].get('parent') is not None:
                events.append(f" Node {s_i} lost connection")
            new_routing_data[s_i] = {
                'parent':      None,
                'alive':       False,
                'dropped':     True,
                'retries':     0,
                'dead_logged': True
            }

    avg_eng = total_energy / NUM_NODES
    avg_sig = total_signal / active_nodes if active_nodes > 0 else 0
    return elements, avg_eng, avg_sig, new_routing_data, events