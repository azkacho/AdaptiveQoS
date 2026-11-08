# Adaptive QoS-Aware Switching Strategy for nRF24L01-Based IoT Networks

A simulation framework for adaptive Quality of Service (QoS) switching in dynamic clustering-tree topologies, built for nRF24L01-based wireless sensor networks.

This repository is part of an undergraduate thesis (Tugas Akhir) in Electronics and Instrumentation at Universitas Gadjah Mada, 2026.

## Overview

This project uses Deep Reinforcement Learning (Deep Q-Network) to address QoS degradation in wireless sensor networks built on nRF24L01 transceivers. The system models a dynamic clustering-tree topology in which an AI agent, acting as a node or cluster head, adaptively switches routing strategy and resource allocation in response to changing network conditions such as interference, traffic density, and battery depletion.

## Features

- Custom RL environment built on the Gymnasium API (`wsn_rl_env.py`)
- DQN agent implemented in PyTorch, trained to optimize throughput, latency, and energy efficiency
- Interactive analytics dashboard built with Plotly Dash (`app.py`) for real-time topology visualization and performance monitoring
- Baseline comparison against conventional Random and Greedy algorithms
- Multi-scenario testing across three experimental conditions (A, B, C)

## Project Structure

```
AdaptiveQoS/
├── analysis_output/
│   └── training_convergence/
│       ├── fig_reward_convergence.png
│       ├── fig_qos_epsilon.png
│       ├── fig_curriculum_edf.png
│       ├── tabel_ranking_model_mcdm.csv
│       └── tabel_training_summary.csv
├── assets/                    # Dashboard CSS
├── components/                # Dashboard UI modules
│   ├── callbacks.py
│   ├── layout.py
│   └── logic.py
├── experiment_data/           # Raw experiment logs (Scenarios A, B, C)
├── logs/                      # Training logs
├── models/                    # Trained PyTorch checkpoints (.pth)
├── results/                   # Output plots and comparison files
├── analysis_training.py
├── app.py                     # Runs the interactive dashboard
├── baseline_random.py         # Baseline algorithm
├── compare_results.py         # Generates AI vs baseline comparison plots
├── config_rl.py                # Global network and RL hyperparameters
├── extract_logs.py
├── find_central_node.py
├── main-rl.py                  # Trains and evaluates the DQN model
├── plot_only.py
├── plot_results.py
├── wsn_rl_env.py                # Gymnasium WSN environment module
└── README.md
```

## Installation

Requires Python 3.8 or later. A virtual environment is recommended.

```bash
git clone https://github.com/azkacho/adaptiveqos.git
cd adaptiveqos
pip install torch pandas gymnasium dash plotly numpy matplotlib
```

## Usage

### Train the model

```bash
python main-rl.py
```

Hyperparameters such as episode count, grid size, and traffic range can be adjusted in `config_rl.py`.

### Run the dashboard

```bash
python app.py
```

Open `http://127.0.0.1:8050/` in your browser. The dashboard renders layouts from `components/` and loads experiment data from `experiment_data/`.

### Compare against baselines

```bash
python baseline_random.py
python compare_results.py
```

Comparison plots are written to `results/`.

## Simulation Scenarios

- **Scenario A** — traffic load variation and its effect on packet delivery ratio and latency
- **Scenario B** — reliability under fluctuating nRF24L01 radio link interference
- **Scenario C** — energy depletion, measuring network lifetime

## Author

Azka Choirul Munna
S1 Electronics and Instrumentation, Universitas Gadjah Mada
Email: azkachoirul03@gmail.com · LinkedIn: https://www.linkedin.com/in/azkachoirul/

Built for undergraduate thesis research. If you use or adapt this method, please cite this repository or the accompanying thesis.
