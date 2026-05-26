import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv

from wsn_rl_env import WSN_RL_Env
from config_rl import (
    NUM_EPISODES, TIMESTEPS_PER_EPISODE, RANDOM_SEED,
    EDF_TRAIN_STRATEGY, EDF_CURRICULUM_PHASES  # [BARU]
)


# --- 1. Neural Network (Q-Network) ---

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


# --- 2. Agen DQN ---

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.memory       = deque(maxlen=10000)
        self.gamma        = 0.99
        self.epsilon      = 1.0
        self.epsilon_min  = 0.05
        self.epsilon_decay = 0.997
        self.learning_rate = 0.0003
        self.batch_size   = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model        = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch  = random.sample(self.memory, self.batch_size)
        states     = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions    = torch.LongTensor(np.array([m[1] for m in minibatch])).unsqueeze(1).to(self.device)
        rewards    = torch.FloatTensor(np.array([m[2] for m in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones      = torch.FloatTensor(np.array([m[4] for m in minibatch])).to(self.device)

        current_q  = self.model(states).gather(1, actions).squeeze(1)
        next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
        next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        expected_q = rewards + (1 - dones) * self.gamma * next_q #Bellmanns equation

        loss = nn.MSELoss()(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# --- 3. Main Loop Pelatihan ---

if __name__ == "__main__":
    env = WSN_RL_Env()

    # state_dim otomatis terbaca = 7 (6 lama + 1 EDF)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"WSN Environment Initialized.")
    print(f"State Dimension : {state_dim}  (termasuk EDF)")
    print(f"Action Dimension: {action_dim}")
    print(f"EDF Strategy    : {EDF_TRAIN_STRATEGY}")
    print(f"Training for {NUM_EPISODES} episodes...\n")

    agent = DQNAgent(state_dim, action_dim)

    history = {'rewards': [], 'avg_qos': [], 'drops': [], 'retries': [], 'edf': [],
               'psr': [], 'steps_taken': []}  # PSR dan steps_taken dicatat per episode

    # Setup direktori output
    for d in ["models", "logs", "results"]:
        os.makedirs(d, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name  = f"wsn_dqn_E{NUM_EPISODES}_T{TIMESTEPS_PER_EPISODE}_{timestamp}"
    model_path = os.path.join("models", f"{base_name}.pth")
    csv_path   = os.path.join("logs",   f"train_log_{base_name}.csv")

    # Header CSV — Steps_Taken dan PSR ditambahkan untuk analisis yang valid
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Episode', 'Reward', 'Avg_QoS', 'Drops', 'Retries', 'Epsilon',
            'EDF_Episode',
            'Steps_Taken',  # langkah aktual per episode (bisa < TIMESTEPS_PER_EPISODE)
            'PSR',          # = 1 - (Drops / Steps_Taken), dihitung di sini bukan di luar
        ])

    for e in range(NUM_EPISODES):
        state, _ = env.reset(seed=RANDOM_SEED + e)
        episode_edf   = env.edf         # [BARU] catat EDF episode ini
        total_reward  = 0
        ep_qos_scores = []
        ep_retries    = 0
        ep_drops      = 0

        steps_taken = 0  # jumlah langkah aktual dalam episode ini

        for t in range(TIMESTEPS_PER_EPISODE):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            drop_occurred = info.get('dropped', False)

            # ── Pastikan drop tidak terminasi episode ────────────────────────────
            # wsn_rl_env.py sudah difix: is_dropped → terminated=False
            # Override ini tetap dipertahankan sebagai safety net — jika env
            # di-update di luar dan tidak sengaja mengembalikan terminated=True
            # pada drop, training loop di sini tetap aman.
            is_true_terminal = terminated and not drop_occurred

            done = is_true_terminal or truncated
            agent.remember(state, action, reward, next_state, is_true_terminal)

            state         = next_state
            total_reward += reward
            steps_taken  += 1

            ep_qos_scores.append(info.get('S_QoS', 0))
            ep_retries   += info.get('retries', 0)
            if drop_occurred:
                ep_drops += 1

            agent.replay()
            if done:
                break

        agent.update_target_model()
        agent.decay_epsilon()

        avg_qos = np.mean(ep_qos_scores) if ep_qos_scores else 0

        # PSR dihitung dari steps aktual, bukan konstanta TIMESTEPS_PER_EPISODE
        # Formula: PSR = 1 - (ep_drops / steps_taken)
        # steps_taken = langkah nyata yang dijalankan (bisa < TIMESTEPS_PER_EPISODE
        # jika true terminal terjadi lebih awal)
        psr = 1.0 - (ep_drops / steps_taken) if steps_taken > 0 else 1.0

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                e + 1, total_reward, avg_qos, ep_drops, ep_retries,
                agent.epsilon, round(episode_edf, 4),
                steps_taken,
                round(psr, 6),
            ])

        history['rewards'].append(total_reward)
        history['avg_qos'].append(avg_qos)
        history['drops'].append(ep_drops)
        history['retries'].append(ep_retries)
        history['edf'].append(episode_edf)
        history['psr'].append(psr)
        history['steps_taken'].append(steps_taken)

        if (e + 1) % 10 == 0:
            print(f"Ep {e+1:4d}/{NUM_EPISODES} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"QoS: {avg_qos:.3f} | "
                  f"Drops: {ep_drops:3d} | "
                  f"PSR: {psr:.4f} | "
                  f"Steps: {steps_taken:4d} | "
                  f"Retries: {ep_retries:3d} | "
                  f"eps: {agent.epsilon:.3f} | "
                  f"EDF: {episode_edf:.2f}")

    # --- 4. Simpan Model & Plot ---
    torch.save(agent.model.state_dict(), model_path)

    plt.figure(figsize=(12, 12))

    plt.subplot(5, 1, 1)
    plt.plot(history['rewards'])
    plt.title('Total Reward per Episode')
    plt.ylabel('Reward')

    plt.subplot(5, 1, 2)
    plt.plot(history['avg_qos'], color='green')
    plt.title('Average QoS Score per Episode')
    plt.ylabel('S_QoS')

    plt.subplot(5, 1, 3)
    plt.plot(history['retries'], color='orange')
    plt.title('Total Retransmissions per Episode')
    plt.ylabel('Retries')

    plt.subplot(5, 1, 4)
    plt.plot(history['psr'], color='green', linewidth=0.8, alpha=0.5)
    # Moving average PSR agar tren lebih jelas
    window = min(10, len(history['psr']))
    psr_smooth = np.convolve(history['psr'], np.ones(window)/window, mode='valid')
    plt.plot(range(window - 1, len(history['psr'])), psr_smooth, color='green', linewidth=2)
    plt.axhline(y=0.95, color='red', linestyle='--', linewidth=0.8, label='PSR 0.95')
    plt.title('Packet Success Rate (PSR) per Episode  —  PSR = 1 - Drops/Steps_Taken')
    plt.ylabel('PSR')
    plt.ylim(0, 1.05)
    plt.legend(fontsize=8)

    # [BARU] Plot EDF per episode untuk verifikasi distribusi training
    plt.subplot(5, 1, 5)
    plt.scatter(range(NUM_EPISODES), history['edf'], s=2, alpha=0.4, color='purple')
    plt.title('EDF per Episode (verifikasi distribusi training)')
    plt.ylabel('EDF')
    plt.xlabel('Episode')
    # Garis batas fase curriculum
    if EDF_TRAIN_STRATEGY == 'curriculum':
        colors_phase = ['#aaaaaa', '#888888', '#555555']
        for (ep_lim, edf_max), c in zip(EDF_CURRICULUM_PHASES, colors_phase):
            plt.axvline(x=ep_lim, color=c, linestyle='--', linewidth=0.8)
            plt.axhline(y=edf_max, color=c, linestyle=':', linewidth=0.6)

    plt.tight_layout()
    plot_path = os.path.join("results", f"plot_{base_name}.png")
    plt.savefig(plot_path)

    print(f"\nTraining selesai!")
    print(f"  PTH : {model_path}")
    print(f"  CSV : {csv_path}")
    print(f"  PNG : {plot_path}")
    plt.show()