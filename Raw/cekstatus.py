import pandas as pd
import numpy as np

# Membaca file CSV
df = pd.read_csv('customer_aggregated.csv')

print("="*80)
print("LAPORAN PEMERIKSAAN STATUS DATA")
print("="*80)

# 1. Informasi Umum Dataset
print("\n1. INFORMASI UMUM DATASET")
print("-"*80)
print(f"Jumlah Baris: {df.shape[0]:,}")
print(f"Jumlah Kolom: {df.shape[1]}")
print(f"Ukuran Dataset: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. Informasi Kolom
print("\n2. INFORMASI KOLOM")
print("-"*80)
print(df.info())

# 3. Tampilan Data
print("\n3. PREVIEW DATA (5 Baris Pertama)")
print("-"*80)
print(df.head())

# 4. Statistik Deskriptif
print("\n4. STATISTIK DESKRIPTIF")
print("-"*80)
print(df.describe())

# 5. Missing Values
print("\n5. MISSING VALUES (Data Kosong)")
print("-"*80)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Kolom': missing.index,
    'Jumlah Missing': missing.values,
    'Persentase (%)': missing_pct.values
})
print(missing_df[missing_df['Jumlah Missing'] > 0])
print(f"\nTotal Missing Values: {df.isnull().sum().sum():,}")

# 6. Duplikasi Data
print("\n6. DUPLIKASI DATA")
print("-"*80)
duplicates = df.duplicated().sum()
print(f"Jumlah Baris Duplikat: {duplicates:,}")
print(f"Persentase Duplikat: {(duplicates/len(df)*100):.2f}%")

# 7. Tipe Data Kolom
print("\n7. TIPE DATA SETIAP KOLOM")
print("-"*80)
for col in df.columns:
    print(f"{col:20} : {df[col].dtype}")

# 8. Nilai Unik untuk Kolom Kategorikal
print("\n8. JUMLAH NILAI UNIK")
print("-"*80)
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col:20} : {unique_count:,} nilai unik")

# 9. Deteksi Outlier pada Kolom Numerik
print("\n9. DETEKSI OUTLIER (Kolom Numerik)")
print("-"*80)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
    print(f"{col:20} : {outliers:,} outlier ({(outliers/len(df)*100):.2f}%)")

# 10. Nilai Negatif (untuk kolom yang seharusnya positif)
print("\n10. CEK NILAI NEGATIF")
print("-"*80)
for col in numeric_cols:
    negative_count = (df[col] < 0).sum()
    if negative_count > 0:
        print(f"{col:20} : {negative_count:,} nilai negatif")

# 11. Distribusi Data (untuk kolom penting)
print("\n11. DISTRIBUSI DATA PENTING")
print("-"*80)
if 'Country' in df.columns:
    print("\nTop 10 Negara:")
    print(df['Country'].value_counts().head(10))

if 'StockCode' in df.columns:
    print(f"\nJumlah Produk Unik: {df['StockCode'].nunique():,}")

if 'Customer ID' in df.columns or 'CustomerID' in df.columns:
    customer_col = 'Customer ID' if 'Customer ID' in df.columns else 'CustomerID'
    print(f"\nJumlah Customer Unik: {df[customer_col].nunique():,}")

# 12. Kesimpulan & Rekomendasi
print("\n" + "="*80)
print("KESIMPULAN & REKOMENDASI")
print("="*80)

rekomendasi = []
if df.isnull().sum().sum() > 0:
    rekomendasi.append("✓ Terdapat missing values yang perlu ditangani")
if duplicates > 0:
    rekomendasi.append("✓ Terdapat data duplikat yang perlu diperiksa")
if any((df[col] < 0).sum() > 0 for col in numeric_cols):
    rekomendasi.append("✓ Terdapat nilai negatif yang perlu divalidasi")

if rekomendasi:
    for r in rekomendasi:
        print(r)
else:
    print("✓ Data dalam kondisi baik!")

print("\n" + "="*80)