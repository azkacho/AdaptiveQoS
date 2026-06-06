# AdaptiveQoS

AdaptiveQoS/
├── analysis_output/training_convergence
│   ├── fig_reward_convergence.png
│   ├── fig_qos_epsilon.png
│   ├── fig_curriculum_edf.png
│   ├── tabel_ranking_model_mcdm.csv
│   ├── tabel_training_summary.csv
├── assets/
│   └── style.css
├── components/
│   ├── _pycache_/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── layout.py
│   └── logic.py
├── experiments_data/
│   ├── Skenario(A_...).csv & txt
│   ├── Skenario(B _...). csv & txt
│   └── Skenario(C_ ...). csv & txt
├── logs/
│   └── train_log_wsn_dqn_E(Episode)_T(Timesteps)_(timestamp).csv
├── models/
│   └── wsn_dqn_E(Episode)_T(Timesteps)(timestamp).pth
├── results/
│   ├── baseline_metrics_E(Episode)_T(Timesteps)_(timestamp).png
│   ├── evaluasi_wsn_E(Episode)_T(Timesteps)_(timestamp).png
│   ├── grafik_perbandingan_wsn_baseline_...png
│   ├── training_metrics_...png
│   ├── baseline_metrics_...png
│   ├── DQN_topology_...png
│   ├── plot_wsn_dqn_...png
│   ├── hasil_evaluasi_wsn_dqn_...csv
│   └── hasil_baseline_random_...csv
├── analysis_training.py
├── app.py
├── baseline_random.py
├── compare_results.py
├── config_rl.py
├── extract_logs.py
├── main-rl.py
├── plot_only.py
├── plot_results.py
└── wsn_rl_env.py



# Daftar Library Yang digunakan dan Dependendsinya

# _____________________________________________________________________________________
[analysis_traianing.py]
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

# _____________________________________________________________________________________
[app.py]
import dash
*from components.layout import create_layout*
*from components.callbacks import *register_callbacks*

# _____________________________________________________________________________________
[baseline_random.py]
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
*from wsn_rl_env import WSN_RL_Env*
from datetime import datetime

# _____________________________________________________________________________________
[compare_results.py]
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
*from config_rl import NUM_EPISODES, TIMESTEPS_PER_EPISODE*

# _____________________________________________________________________________________
[find_central_node.py]
import sys, os, random, collections, glob
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
*from wsn_rl_env import WSN_RL_Env*
*from config_rl import RANDOM_SEED, SOURCE_NODES_COUNT*

# _____________________________________________________________________________________
[main-rl.py]
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

# _____________________________________________________________________________________
[plot_only.py]
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# _____________________________________________________________________________________
[plot_result.py]
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")
search_pattern = os.path.join(results_dir, "hasil_evaluasi_wsn_dqn_E1000_T1000_*.csv")
list_of_files = glob.glob(search_pattern)

# _____________________________________________________________________________________
[wsn_rl_env.py]
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import random
import math

*from config_rl import (*
    NUM_NODES, AREA_SIZE, MAX_COMM_DISTANCE, SINK_POSITION,
    DATA_RATE_THRESHOLDS, MAX_BUFFER_CAPACITY,
    POWER_CONSUMPTION, NUM_PARENT_OPTIONS,
    PACKET_SIZE_BITS, PACKET_ARRIVAL_RATE, TIMESTEPS_PER_EPISODE,
    MAX_RETRANSMISSIONS, PENALTY_DROP, ALPHA, BETA,
    QOS_WEIGHTS, MAX_EXPECTED_LATENCY, INITIAL_ENERGY_JOULE, RANDOM_SEED,
    EDF_TRAIN_MAX, EDF_TRAIN_STRATEGY, EDF_CURRICULUM_PHASES
*)*

# _____________________________________________________________________________________