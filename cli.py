import os
import pandas as pd
from src.standardization.services import process_dataset
from src.visualization.services import auto_select_k, run_clustering
from src.config import DATA_DIR


def main():
    while True:
        print("\n=== MENU UTAMA ===")
        print("1. Standardization")
        print("2. Clustering")
        print("3. Keluar")
        choice = input("Pilih [1-3]: ").strip()

        if choice == "1":
            file_name = input("Masukkan nama file (CSV/XLSX) di folder datasets/: ").strip()
            file_path = os.path.join("datasets", file_name)
            try:
                result = process_dataset(file_path)
                print(f"✅ Standardisasi selesai: {result['standardized_file']}")
            except Exception as e:
                print(f"❌ Error: {e}")

        elif choice == "2":
            files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("standardized_data")], reverse=True)
            if not files:
                print("❌ Tidak ada file standardized. Jalankan menu 1 dulu.")
                continue
            df = pd.read_csv(os.path.join(DATA_DIR, files[0]))

            mode = input("Pilih mode K (1=Manual, 2=Auto): ").strip()
            if mode == "1":
                k = int(input("Masukkan jumlah cluster: "))
            else:
                k, _ = auto_select_k(df)

            master_files = sorted([f for f in os.listdir(DATA_DIR) if f.startswith("dataset_master")], reverse=True)
            df_master = pd.read_csv(os.path.join(DATA_DIR, master_files[0]))

            try:
                result = run_clustering(df, df_master, k)
                print(result["profile"])
            except ValueError as e:
                print(f"❌ Error: {str(e)}")

        elif choice == "3":
            break

        else:
            print("⚠️ Pilihan tidak valid.")


if __name__ == "__main__":
    main()
