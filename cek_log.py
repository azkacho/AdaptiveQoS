import os

# Ganti dengan path absolute folder logs Anda
path_log = r"D:\UGM\TA\AdaptiveQoS\results"

found = False
print(f"Memeriksa isi folder: {path_log}")

for root, dirs, files in os.walk(path_log):
    for file in files:
        if "tfevents" in file:
            print(f"\n LOG DITEMUKAN!")
            print(f"Lokasi: {root}")
            print(f"Nama File: {file}")
            found = True

if not found:
    print("\n TIDAK ADA LOG.")
    print("Pastikan di main-rl.py parameter 'tensorboard_log' sudah benar.")