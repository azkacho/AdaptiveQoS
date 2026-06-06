Rencana Perbaikan Naskah Skripsi

Judul: SWITCHING ADAPTIF BERBASIS QOS PADA TOPOLOGI CLUSTERING TREE DINAMIS JARINGAN IOT BERBASIS NRF24L01
Penulis: Azka Choirul Munna
Model RL Terbaik (Empiris): DQN - Episode 500, Timestep 1000 (E500 T1000)

Dokumen ini memuat daftar perbaikan (Action Plan) berdasarkan peninjauan draf .docx dan arsitektur kode dari repository Github. Beri tanda centang [x] jika bagian tersebut sudah direvisi di Microsoft Word.

BAB II: TINJAUAN PUSTAKA

[ ] Revisi Subbab 2.5 / 2.10 (Integrasi MCDM dalam Evaluasi RL)

Tujuan: Memberikan landasan teori mengapa pemilihan model Reinforcement Learning di WSN memerlukan pendekatan Multi-Criteria Decision Making (MCDM).

Poin Narasi: - RL pada WSN dengan environment yang stokastik tidak bisa hanya dinilai dari Cumulative Reward tertinggi (karena berpotensi overfitting atau tidak stabil).

Menjelaskan bahwa parameter seperti variansi loss dan kecepatan konvergensi sama pentingnya dengan pencapaian reward.

Menjelaskan konsep dasar pembobotan kriteria (MCDM) untuk memilih model yang paling robust sebagai baseline agent untuk di-deploy ke sistem.

Target Aksi: Menambahkan 2-3 paragraf teori baru dan memasukkan 1-2 referensi ilmiah terkait hibridisasi RL dan MCDM atau evaluasi multikriteria pada model AI.

BAB III: METODOLOGI PENELITIAN

[ ] Revisi Subbab 3.6 (Prosedur Pelatihan Agen & Eksplorasi)

Tujuan: Mendeskripsikan secara matematis dan logis bagaimana agen DQN belajar mengenali lingkungan nRF24L01.

Poin Narasi:

Memasukkan subbagian khusus tentang Epsilon-Greedy Strategy.

Fase Eksplorasi ($\epsilon$ mendekati 1): Jelaskan mengapa agen butuh mengambil tindakan acak di awal (untuk memetakan kualitas link dan state-space topologi tree yang dinamis).

Fase Eksploitasi & Epsilon Decay: Masukkan persamaan peluruhan epsilon ($\epsilon_{t+1} = \epsilon_t \times decay\_rate$) dan kapan agen mulai memanfaatkan Q-Table/DQN untuk memilih rute terbaik.

[ ] Revisi Subbab 3.8 (Skenario Pengujian & Pendekatan MCDM)

Tujuan: Menjadikan mekanisme ekstraksi log (analysis_training.py) sebagai metodologi ilmiah formal, bukan sekadar "pemilihan manual".

Poin Narasi:

Jelaskan alur evaluasi: Mulai dari ekstraksi log CSV pelatihan -> perhitungan normalisasi metrik (reward, epsilon konvergensi, loss) -> perankingan MCDM.

Menegaskan bahwa pendekatan ini memberikan justifikasi kuantitatif yang objektif untuk memilih model E500 T1000.

BAB IV: HASIL DAN PEMBAHASAN

[ ] Penambahan Subbab 4.2.4 (Hasil Seleksi Model - MCDM)

Tujuan: Membuktikan secara data mengapa E500 T1000 adalah model terbaik.

Poin Narasi:

Menampilkan tabel dari tabel_ranking_model_mcdm.csv ke dalam format tabel skripsi yang standar (APA/IEEE style).

Membahas fenomena mengapa model dengan E1000 atau E2000 justru memiliki skor lebih rendah (indikasi overfitting terhadap skenario awal atau penalti akibat variansi loss yang tinggi).

Klaim konklusi pemilihan model E500 T1000 sebelum masuk ke tahap pengujian Digital Twin.

[ ] Revisi Subbab 4.4 (Penyederhanaan Narasi Skenario Digital Twin)

Tujuan: Mengaitkan simulasi visual app.py (Dash Plotly) dengan narasi pengujian yang runut dan mudah dipahami.

Poin Narasi:

Pecah penjelasan menjadi tiga skenario definitif berdasarkan eksperimen:

Skenario A: Kondisi Normal (Baseline comparison antara AI vs Greedy/Random).

Skenario B: Kondisi Gangguan (Node Failure / Node mati tiba-tiba).

Skenario C: Kondisi Fluktuasi Kualitas Tautan (Interference / Congestion jaringan).

Jelaskan peran Dashboard Digital Twin sebagai antarmuka monitoring real-time yang mensimulasikan data telemetri WSN sebenarnya.

[ ] Revisi Subbab 4.5 (Restrukturisasi Metrik Evaluasi)

Tujuan: Memisahkan metrik AI (backend) dengan metrik WSN (frontend/operational).

Poin Narasi: Buat hierarki yang jelas:

Metrik Pelatihan (DQN): Cumulative Reward, Epsilon, Loss.

Metrik Seleksi (MCDM): Stabilitas reward, kecepatan konvergensi (Episode converge).

Metrik Operasional WSN: PDR (Packet Delivery Ratio), End-to-End Latency (jika ada), Konsumsi Energi (estimasi), dan Throughput. (Tampilkan perbandingannya berdasarkan skenario di Subbab 4.4).

[ ] Penghapusan Subbab 4.6 (Tiadakan Potongan Kode Mentah)

Tujuan: Menjaga standar akademis penulisan naskah (menghindari isi skripsi yang dipenuhi source code berlembar-lembar).

Poin Narasi:

Hapus subbab yang isinya murni copy-paste dari .py.

Ganti dengan representasi Pseudocode untuk algoritma yang esensial (misalnya logika switching rute di fungsi step RL).

Parameter simulasi dari config_rl.py diubah bentuknya menjadi Tabel Parameter Konfigurasi dan diletakkan di Bab 3 (Metodologi) atau awal Bab 4.

Catatan Pembimbing: Selalu diskusikan draf narasi sebelum dimasukkan sepenuhnya ke file utama Word.