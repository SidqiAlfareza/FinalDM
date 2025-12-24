import pandas as pd
import os

# Path file
xlsx_file = "online_retail_II.xlsx"
csv_file = "online_retail_II.csv"

# Baca file Excel
print(f"Membaca file {xlsx_file}...")
df = pd.read_excel(xlsx_file)

# Simpan ke CSV dengan separator koma
print(f"Mengkonversi ke {csv_file}...")
df.to_csv(csv_file, index=False, sep=',')

print(f"Konversi berhasil! File CSV disimpan di: {os.path.abspath(csv_file)}")
print(f"Jumlah baris: {len(df)}")
print(f"Jumlah kolom: {len(df.columns)}")
