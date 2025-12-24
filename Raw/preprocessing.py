import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt 
import seaborn as sns

class RetailDataPreprocessor:
    """
    Class untuk preprocessing data Online Retail II
    
    Fitur-fitur preprocessing yang diimplementasikan:
    1. Handle Missing Values - Menangani nilai yang hilang
    2. Remove Duplicates - Menghapus data duplikat
    3. Handle Negative Values - Menangani nilai negatif (return/cancellation)
    4. Data Type Conversion - Konversi tipe data yang sesuai
    5. Feature Engineering - Membuat fitur baru
    6. Outlier Detection - Deteksi dan penanganan outlier
    7. Data Validation - Validasi konsistensi data
    """
    
    def __init__(self, filepath):
        """Inisialisasi dengan membaca file CSV"""
        print("="*60)
        print("MEMUAT DATA")
        print("="*60)
        self.df_original = pd.read_csv(filepath)
        self.df = self.df_original.copy()
        print(f"Data berhasil dimuat: {self.df.shape[0]} baris, {self.df.shape[1]} kolom")
        print("\nKolom-kolom dalam dataset:")
        print(self.df.columns.tolist())
        
    def show_initial_info(self):
        """Menampilkan informasi awal dataset"""
        print("\n" + "="*60)
        print("INFORMASI AWAL DATASET")
        print("="*60)
        print(f"\nJumlah baris: {self.df.shape[0]}")
        print(f"Jumlah kolom: {self.df.shape[1]}")
        print(f"\nTipe data:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\nStatistik deskriptif:")
        print(self.df.describe())
        
    def handle_missing_values(self):
        """
        FITUR 1: Handle Missing Values
        - Hapus baris dengan missing Invoice, StockCode, dan Customer ID
        - Isi missing Description dengan 'Unknown'
        - Isi missing Country dengan 'Unknown'
        """
        print("\n" + "="*60)
        print("FITUR 1: MENANGANI MISSING VALUES")
        print("="*60)
        
        missing_before = self.df.isnull().sum()
        print("\nMissing values sebelum preprocessing:")
        print(missing_before[missing_before > 0])
        
        # Hapus baris dengan missing values di kolom krusial
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['Invoice', 'StockCode', 'Customer ID'])
        removed = initial_rows - len(self.df)
        print(f"\n- Dihapus {removed} baris dengan missing Invoice/StockCode/Customer ID")
        
        # Isi missing Description
        if self.df['Description'].isnull().sum() > 0:
            missing_desc = self.df['Description'].isnull().sum()
            self.df['Description'].fillna('Unknown', inplace=True)
            print(f"- {missing_desc} missing Description diisi dengan 'Unknown'")
        else:
            print(f"- Tidak ada missing Description")
        
        # Isi missing Country
        if self.df['Country'].isnull().sum() > 0:
            missing_country = self.df['Country'].isnull().sum()
            self.df['Country'].fillna('Unknown', inplace=True)
            print(f"- {missing_country} missing Country diisi dengan 'Unknown'")
        else:
            print(f"- Tidak ada missing Country")
        
        missing_after = self.df.isnull().sum()
        print(f"\nMissing values setelah preprocessing:")
        if missing_after.sum() > 0:
            print(missing_after[missing_after > 0])
        else:
            print("Tidak ada missing values")
        
    def remove_duplicates(self):
        """
        FITUR 2: Remove Duplicates
        - Menghapus baris yang benar-benar duplikat
        """
        print("\n" + "="*60)
        print("FITUR 2: MENGHAPUS DUPLIKAT")
        print("="*60)
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_rows - len(self.df)
        print(f"\n- Dihapus {duplicates_removed} baris duplikat")
        print(f"- Jumlah baris sekarang: {len(self.df)}")
        
    def handle_negative_values(self):
        """
        FITUR 3: Handle Negative Values
        - Hapus baris yang memiliki Quantity < 0 (cancellation/return)
        """
        print("\n" + "="*60)
        print("FITUR 3: MENANGANI NILAI NEGATIF")
        print("="*60)
        
        negative_quantity = (self.df['Quantity'] < 0).sum()
        
        print(f"\n- Jumlah transaksi dengan Quantity negatif (cancellation/return): {negative_quantity}")
        
        # Hapus baris dengan Quantity < 0
        initial_rows = len(self.df)
        self.df = self.df[self.df['Quantity'] > 0]
        removed = initial_rows - len(self.df)
        print(f"- Dihapus {removed} baris dengan Quantity negatif")
        print(f"- Jumlah baris sekarang: {len(self.df)}")
        
    def convert_data_types(self):
        """
        FITUR 4: Data Type Conversion
        - Konversi InvoiceDate ke datetime
        - Pastikan tipe data numerik untuk Quantity (int64) dan Price (float64)
        - Customer ID = Object
        """
        print("\n" + "="*60)
        print("FITUR 4: KONVERSI TIPE DATA")
        print("="*60)
        
        print("\nTipe data sebelum konversi:")
        print(self.df.dtypes)
        
        # Konversi InvoiceDate ke datetime
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        print("\n- InvoiceDate dikonversi ke datetime")
        
        # Konversi Quantity ke int64
        self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce').astype('int64')
        print("- Quantity dikonversi ke int64")
        
        # Konversi Price ke float64
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce').astype('float64')
        print("- Price dikonversi ke float64")
        
        # Customer ID tetap sebagai Object (string)
        self.df['Customer ID'] = self.df['Customer ID'].astype(str)
        print("- Customer ID dikonversi ke object (string)")
        
        print("\nTipe data setelah konversi:")
        print(self.df.dtypes)
        
    def feature_engineering(self):
        """
        FITUR 5: Feature Engineering
        - TotalAmount: Quantity * Price (total nilai transaksi)
        - Year, Month, Day, Hour, DayOfWeek dari InvoiceDate
        - IsWeekend: Apakah transaksi di weekend
        """
        print("\n" + "="*60)
        print("FITUR 5: FEATURE ENGINEERING")
        print("="*60)
        
        # Total Amount
        self.df['TotalAmount'] = self.df['Quantity'] * self.df['Price']
        print("\n- TotalAmount: Quantity × Price")
        
        # Ekstraksi fitur waktu
        self.df['Year'] = self.df['InvoiceDate'].dt.year
        self.df['Month'] = self.df['InvoiceDate'].dt.month
        self.df['Day'] = self.df['InvoiceDate'].dt.day
        self.df['Hour'] = self.df['InvoiceDate'].dt.hour
        self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
        print("- Ekstraksi fitur waktu: Year, Month, Day, Hour, DayOfWeek")
        
        # Is Weekend (Saturday=5, Sunday=6)
        self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6])
        print("- IsWeekend: Menandai transaksi di weekend (Saturday & Sunday)")
        
        print(f"\nFitur baru yang dibuat:")
        new_features = ['TotalAmount', 'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'IsWeekend']
        for feature in new_features:
            print(f"  • {feature}")
        
    def detect_outliers(self):
        """
        FITUR 6: Outlier Detection
        - Deteksi outlier menggunakan IQR method
        - Tampilkan statistik outlier
        - Opsi: Remove atau keep outlier
        """
        print("\n" + "="*60)
        print("FITUR 6: DETEKSI OUTLIER")
        print("="*60)
        
        # Deteksi outlier untuk Quantity, Price, dan TotalAmount
        numeric_cols = ['Quantity', 'Price', 'TotalAmount']
        
        outlier_summary = {}
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(self.df)) * 100
            
            print(f"\n{col}:")
            print(f"  • Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"  • Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
            print(f"  • Jumlah outlier: {outliers} ({outlier_pct:.2f}%)")
            
            # Simpan informasi outlier
            self.df[f'{col}_IsOutlier'] = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_summary[col] = outliers
        
        print("\n" + "-"*60)
        print("KEPUTUSAN OUTLIER:")
        print("-"*60)
        print("- Kolom flag outlier ditambahkan: Quantity_IsOutlier, Price_IsOutlier, TotalAmount_IsOutlier")
        print("- CATATAN: Outlier TIDAK dihapus, hanya ditandai untuk analisis lebih lanjut")
        print("- Outlier dapat digunakan untuk analisis pola pembelian bulk atau transaksi khusus")
        
    def validate_data(self):
        """
        FITUR 7: Data Validation
        - Validasi konsistensi data
        - Cek nilai tidak valid
        """
        print("\n" + "="*60)
        print("FITUR 7: VALIDASI DATA")
        print("="*60)
        
        # Validasi TotalAmount calculation
        calculated_total = self.df['Quantity'] * self.df['Price']
        mismatch = ~np.isclose(self.df['TotalAmount'], calculated_total, rtol=1e-5)
        print(f"\n- Validasi TotalAmount: {mismatch.sum()} ketidakcocokan")
        
        if mismatch.sum() > 0:
            print("  PERINGATAN: Ada ketidakcocokan dalam perhitungan TotalAmount!")
        else:
            print("  ✓ Semua perhitungan TotalAmount konsisten")
        
        # Cek nilai yang tidak valid
        invalid_quantity = (self.df['Quantity'] <= 0).sum()
        invalid_price = (self.df['Price'] <= 0).sum()
        invalid_total = (self.df['TotalAmount'] <= 0).sum()
        
        print(f"\n- Validasi nilai tidak valid:")
        print(f"  • Quantity <= 0: {invalid_quantity}")
        print(f"  • Price <= 0: {invalid_price}")
        print(f"  • TotalAmount <= 0: {invalid_total}")
        
        if invalid_quantity == 0 and invalid_price == 0 and invalid_total == 0:
            print("  ✓ Tidak ada nilai tidak valid")
        
        # Cek nilai ekstrem
        print(f"\n- Range nilai:")
        print(f"  • Quantity: {self.df['Quantity'].min()} - {self.df['Quantity'].max()}")
        print(f"  • Price: ${self.df['Price'].min():.2f} - ${self.df['Price'].max():.2f}")
        print(f"  • TotalAmount: ${self.df['TotalAmount'].min():.2f} - ${self.df['TotalAmount'].max():.2f}")
        
        # Cek jumlah unik
        print(f"\n- Jumlah nilai unik:")
        print(f"  • Invoice unik: {self.df['Invoice'].nunique():,}")
        print(f"  • Customer unik: {self.df['Customer ID'].nunique():,}")
        print(f"  • StockCode unik: {self.df['StockCode'].nunique():,}")
        print(f"  • Country unik: {self.df['Country'].nunique()}")
        
        # Cek periode data
        print(f"\n- Periode data:")
        print(f"  • Tanggal awal: {self.df['InvoiceDate'].min()}")
        print(f"  • Tanggal akhir: {self.df['InvoiceDate'].max()}")
        print(f"  • Rentang waktu: {(self.df['InvoiceDate'].max() - self.df['InvoiceDate'].min()).days} hari")
        
    def generate_summary_report(self):
        """Generate ringkasan preprocessing"""
        print("\n" + "="*60)
        print("RINGKASAN PREPROCESSING")
        print("="*60)
        
        print(f"\nData Original:")
        print(f"  • Jumlah baris: {self.df_original.shape[0]:,}")
        print(f"  • Jumlah kolom: {self.df_original.shape[1]}")
        
        print(f"\nData Setelah Preprocessing:")
        print(f"  • Jumlah baris: {self.df.shape[0]:,}")
        print(f"  • Jumlah kolom: {self.df.shape[1]}")
        
        rows_removed = self.df_original.shape[0] - self.df.shape[0]
        rows_removed_pct = (rows_removed / self.df_original.shape[0]) * 100
        print(f"  • Baris dihapus: {rows_removed:,} ({rows_removed_pct:.2f}%)")
        print(f"  • Kolom ditambahkan: {self.df.shape[1] - self.df_original.shape[1]}")
        
        print(f"\nKolom baru yang ditambahkan:")
        new_cols = set(self.df.columns) - set(self.df_original.columns)
        for col in sorted(new_cols):
            print(f"  • {col}")
        
        print(f"\nStatistik akhir:")
        print(f"  • Periode transaksi: {self.df['InvoiceDate'].min()} s/d {self.df['InvoiceDate'].max()}")
        print(f"  • Total transaksi: {len(self.df):,}")
        print(f"  • Total revenue: ${self.df['TotalAmount'].sum():,.2f}")
        print(f"  • Average transaction: ${self.df['TotalAmount'].mean():.2f}")
        print(f"  • Median transaction: ${self.df['TotalAmount'].median():.2f}")
        
    def save_preprocessed_data(self, output_path):
        """Simpan data yang sudah dipreprocess"""
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Data berhasil disimpan ke: {output_path}")
        print(f"✓ Total baris tersimpan: {len(self.df):,}")
        
    def run_full_preprocessing(self, output_path='preprocessed_data.csv'):
        """Jalankan semua tahap preprocessing"""
        self.show_initial_info()
        self.handle_missing_values()
        self.remove_duplicates()
        self.handle_negative_values()
        self.convert_data_types()
        self.feature_engineering()
        self.detect_outliers()
        self.validate_data()
        self.generate_summary_report()
        self.save_preprocessed_data(output_path)
        
        return self.df


# ============================================================================
# CARA PENGGUNAAN
# ============================================================================

if __name__ == "__main__":
    # Path file input
    input_file = "online_retail_II.csv"
    output_file = "online_retail_II_preprocessed.csv"
    
    # Inisialisasi preprocessor
    preprocessor = RetailDataPreprocessor(input_file)
    
    # Jalankan preprocessing lengkap
    df_clean = preprocessor.run_full_preprocessing(output_file)
    
    # Tampilkan sample data hasil preprocessing
    print("\n" + "="*60)
    print("SAMPLE DATA HASIL PREPROCESSING (5 baris pertama)")
    print("="*60)
    print(df_clean.head())
    
    print("\n" + "="*60)
    print("INFO DATA HASIL PREPROCESSING")
    print("="*60)
    print(df_clean.info())
