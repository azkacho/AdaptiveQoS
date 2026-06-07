📡 Adaptive QoS-Aware Switching Strategyin Dynamic Clustering Tree Topology for nRF24L01-Based IoT NetworksRepositori ini adalah bagian dari Penelitian Tugas Akhir S1 Elektronika dan Instrumentasi, Universitas Gadjah Mada (2026).

📖 Abstrak / OverviewProyek ini menyediakan sistem simulasi berbasis Deep Reinforcement Learning (Deep Q-Network/DQN) untuk memecahkan masalah degradasi Quality of Service (QoS) pada Jaringan Sensor Nirkabel (WSN) yang menggunakan transceiver nRF24L01.Sistem ini memodelkan topologi Clustering Tree yang dinamis, di mana agen AI (Node/Cluster Head) secara adaptif melakukan switching strategi routing dan alokasi resource berdasarkan kondisi lingkungan jaringan yang berubah-ubah (misalnya interferensi, kepadatan traffic, atau penurunan daya baterai).

✨ Fitur Utama🧠 Custom RL Environment: Lingkungan simulasi WSN kustom yang dibangun di atas API gymnasium (wsn_rl_env.py).
🤖 Agen DQN Cerdas: Menggunakan PyTorch untuk melatih agen dalam mengambil keputusan switching yang memaksimalkan metrik QoS (Throughput, Latency, Energy Efficiency).
📊 Dashboard Analitik Interaktif: Dibangun dengan Plotly Dash (app.py), memungkinkan pengguna untuk memantau performa jaringan, visualisasi topologi, dan metrik secara real-time.
📈 Komparasi Baseline: Dilengkapi dengan algoritma konvensional (Random & Greedy) untuk evaluasi komparatif (baseline_random.py, compare_results.py).📂 Multi-Skenario Pengujian: Modul pemrosesan log untuk menganalisis data eksperimen dari berbagai kondisi (Skenario A, B, dan C).

🗃️  AdaptiveQoS
 ┣ 📂 analysis_output/training_convergence
    ┗  🖼️ fig_reward_convergence.png
    ┗  🖼️ fig_qos_epsilon.png
    ┗  🖼️ fig_curriculum_edf.png
    ┗  📋 tabel_ranking_model_mcdm.csv
    ┗  📋 tabel_training_summary.csv    
 ┣ 📂 assets/                # File CSS untuk styling dashboard Dash
     ┗ 📜 *style.css*  
 ┣ 📂 components/            # Modul UI/UX Dashboard (callbacks.py, layout.py, logic.py)
     ┗  __init.py__
     ┗ 🇵🇾 *callbacks.py* 
     ┗ 🇵🇾 *layout.py*
     ┗ 🇵🇾 *logic.py* 
 ┣ 📂 experiment_data/       # Kumpulan raw data log (Skenario A, B, C)
    ┗   fig_reward_convergence.png
 ┣ 📂 logs/                  # File log hasil training model
    ┗  🌱 train_log_wsn_dqn_E200_T1000.csv
    ┗  🌱 train_log_wsn_dqn_E500_T1000.csv
    ┗  🌱 train_log_wsn_dqn_E700_T1000.csv
    ┗  🌱 train_log_wsn_dqn_E1000_T200.csv
    ┗  🌱 train_log_wsn_dqn_E1000_T500.csv
    ┗  🌱 train_log_wsn_dqn_E1000_T1000.csv
    ┗  🌱 train_log_wsn_dqn_E2000_T1500.csv
 ┣ 📂 models/                # Checkpoint model PyTorch (.pth) yang telah dilatih
    ┗  ✳️ wsn_dqn_E200_T1000.pth
    ┗  ✳️ wsn_dqn_E500_T1000.pth
    ┗  ✳️ wsn_dqn_E700_T1000.pth
    ┗  ✳️ wsn_dqn_E1000_T200.pth
    ┗  ✳️ wsn_dqn_E1000_T500.pth
    ┗  ✳️ wsn_dqn_E1000_T1000.pth
    ┗  ✳️ wsn_dqn_E2000_T1500.pth
 ┣ 📂 results/               # Output grafik (.png) dan file hasil perbandingan (.csv)
 ┣ 🇵🇾 *analysis_training.py*  
 ┣ 🇵🇾 *app.py*                 # 🚀 Skrip utama untuk menjalankan Dashboard Interaktif
 ┣ 🇵🇾 *baseline_random.py*     # Algoritma pembanding (Baseline)
 ┣ 🇵🇾 *compare_results.py*     # Skrip generator grafik komparasi AI vs Baseline
 ┣ 🇵🇾 *config_rl.py*           # Parameter dan Hyperparameter global jaringan & RL
 ┣ 🇵🇾 *extract_logs.py*
 ┣ 🇵🇾 *find_central_node.py*     
 ┣ 🇵🇾 *main-rl.py*             # 🚀 Skrip utama untuk Training & Evaluasi Model DQN
 ┣ 🇵🇾 *plot_only.py*  
 ┣ 🇵🇾 *plot_results.py*  
 ┣ 🇵🇾 *wsn_rl_env.py*          # Modul Environment Gymnasium WSN
 ┗ 📜 *README.md*              # Dokumentasi proyek

📟
✳️
🌱
🖼️
📑


⚙️ Instalasi dan PersiapanPastikan Anda memiliki Python 3.8 atau lebih baru. Disarankan menggunakan Virtual Environment.Clone Repositorigit clone [https://github.com/azkacho/adaptiveqos.git](https://github.com/azkacho/adaptiveqos.git)
cd adaptiveqos

Install DependensiJalankan perintah berikut untuk menginstal library yang dibutuhkan:pip install torch pandas gymnasium dash plotly numpy matplotlib
🚀 Cara Menjalankan Program1. Melatih Model AI (Training DQN)Untuk memulai proses pelatihan agen RL dalam lingkungan WSN, jalankan:python main-rl.py

Catatan: Hyperparameter seperti jumlah EPISODES, ukuran grid, dan rentang traffic dapat diubah melalui file config_rl.py.2. Membuka Dashboard Analitik InteraktifUntuk memvisualisasikan hasil, topologi jaringan, dan performa QoS, jalankan web dashboard:python app.py

Buka browser Anda dan akses http://127.0.0.1:8050/. Dashboard ini akan merender tata letak dari folder components/ dan memuat log data dari folder experiment_data/.3. Komparasi Performa (Evaluasi)Untuk membandingkan hasil strategi AI dengan strategi Baseline (Random/Greedy) dan mencetak grafik .png ke folder results/, jalankan:python baseline_random.py
python compare_results.py

🧪 Skenario SimulasiProyek ini memvalidasi keandalan algoritma melalui beberapa skenario data (dapat ditemukan di folder experiment_data/):Skenario A: Pengujian variasi traffic load dan dampaknya terhadap packet delivery ratio (PDR) dan latensi.Skenario B: Evaluasi keandalan terhadap interferensi link radio nRF24L01 secara fluktuatif.Skenario C: Skenario penipisan energi (Energy Depletion) untuk mengukur umur jaringan (Network Lifetime).👨‍💻 PenulisAzka Choirul Munna Program Studi S1 Elektronika dan InstrumentasiUniversitas Gadjah Mada📧 Kontak: [Email Anda] | 🌐 LinkedIn: [Tautan Profil Anda]Dibuat untuk keperluan Penelitian Skripsi. Penggunaan atau replikasi metode diharapkan mencantumkan sitasi pada karya tulis/repositori ini.