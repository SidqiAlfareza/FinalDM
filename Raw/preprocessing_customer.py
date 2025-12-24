import pandas as pd
import numpy as np

class CustomerDataPreprocessor:
    """
    Class untuk preprocessing data Customer Aggregated
    
    Fitur preprocessing:
    1. Load data customer_aggregated.csv
    2. Konversi CustomerID dari float ke object/string
    3. Validasi dan cleaning data
    """
    
    def __init__(self, filepath):
        """Inisialisasi dengan membaca file CSV"""
        print("="*60)
        print("MEMUAT DATA CUSTOMER")
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
        
    def convert_customer_id_to_object(self):
        """
        FITUR: Konversi CustomerID ke Object/String
        - Mengubah tipe data CustomerID dari float ke object/string
        - Handle missing values di CustomerID
        """
        print("\n" + "="*60)
        print("KONVERSI CUSTOMER ID KE OBJECT")
        print("="*60)
        
        print(f"\nTipe data CustomerID sebelum: {self.df['CustomerID'].dtype}")
        
        # Handle missing values
        missing_before = self.df['CustomerID'].isnull().sum()
        if missing_before > 0:
            print(f"Missing values di CustomerID: {missing_before}")
            self.df = self.df.dropna(subset=['CustomerID'])
            print(f"Baris dengan missing CustomerID dihapus")
        
        # Konversi float ke integer terlebih dahulu (hapus .0)
        self.df['CustomerID'] = self.df['CustomerID'].astype('int64')
        
        # Konversi ke object/string
        self.df['CustomerID'] = self.df['CustomerID'].astype('object')
        
        print(f"Tipe data CustomerID setelah: {self.df['CustomerID'].dtype}")
        print(f"\nContoh CustomerID setelah konversi:")
        print(self.df['CustomerID'].head(10))
        
    def validate_data(self):
        """
        FITUR: Validasi Data
        - Cek apakah ada nilai negatif
        - Cek apakah ada nilai yang tidak logis
        """
        print("\n" + "="*60)
        print("VALIDASI DATA")
        print("="*60)
        
        # Cek nilai negatif
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print("\nMengecek nilai negatif pada kolom numerik:")
        for col in numeric_cols:
            negative_count = (self.df[col] < 0).sum()
            if negative_count > 0:
                print(f"- {col}: {negative_count} nilai negatif")
            else:
                print(f"- {col}: OK (tidak ada nilai negatif)")
        
        # Cek nilai zero pada TotalSpending
        if 'TotalSpending' in self.df.columns:
            zero_spending = (self.df['TotalSpending'] == 0).sum()
            if zero_spending > 0:
                print(f"\nPeringatan: {zero_spending} customer dengan TotalSpending = 0")
        
        # Cek duplikat CustomerID
        duplicates = self.df['CustomerID'].duplicated().sum()
        if duplicates > 0:
            print(f"\nPeringatan: {duplicates} CustomerID duplikat ditemukan")
        else:
            print(f"\nCustomerID: Tidak ada duplikat")
            
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        """
        FITUR: Deteksi dan Hapus Outlier menggunakan IQR
        - Menggunakan metode IQR (Interquartile Range)
        """
        print("\n" + "="*60)
        print("DETEKSI DAN PENANGANAN OUTLIER (IQR)")
        print("="*60)
        
        if columns is None:
            columns = ['TotalSpending', 'TotalTransaction', 'TotalQuantity']
        
        initial_rows = len(self.df)
        
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outliers_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                print(f"\n{col}:")
                print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
                print(f"  Batas: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print(f"  Outliers: {outliers_count}")
                
                # Hapus outliers
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        removed = initial_rows - len(self.df)
        print(f"\nTotal baris dihapus karena outlier: {removed}")
        print(f"Baris tersisa: {len(self.df)}")
        
    def show_final_info(self):
        """Menampilkan informasi akhir dataset setelah preprocessing"""
        print("\n" + "="*60)
        print("INFORMASI AKHIR DATASET")
        print("="*60)
        print(f"\nJumlah baris: {self.df.shape[0]}")
        print(f"Jumlah kolom: {self.df.shape[1]}")
        print(f"\nTipe data:")
        print(self.df.dtypes)
        print(f"\nMissing values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("Tidak ada missing values")
        print(f"\nStatistik deskriptif:")
        print(self.df.describe())
        
    def save_preprocessed_data(self, output_filepath):
        """Menyimpan data yang sudah dipreprocessing"""
        print("\n" + "="*60)
        print("MENYIMPAN DATA")
        print("="*60)
        self.df.to_csv(output_filepath, index=False)
        print(f"Data berhasil disimpan ke: {output_filepath}")
        print(f"Total baris: {len(self.df)}")
        
    def run_full_preprocessing(self, output_filepath=None, remove_outliers=True):
        """
        Menjalankan semua tahapan preprocessing secara berurutan
        
        Parameters:
        - output_filepath: path untuk menyimpan hasil (optional)
        - remove_outliers: True jika ingin menghapus outlier
        """
        print("\n" + "="*60)
        print("MEMULAI PREPROCESSING CUSTOMER DATA")
        print("="*60)
        
        # Tampilkan info awal
        self.show_initial_info()
        
        # Konversi CustomerID ke object
        self.convert_customer_id_to_object()
        
        # Validasi data
        self.validate_data()
        
        # Hapus outliers (optional)
        if remove_outliers:
            self.remove_outliers_iqr()
        
        # Tampilkan info akhir
        self.show_final_info()
        
        # Simpan hasil
        if output_filepath:
            self.save_preprocessed_data(output_filepath)
        
        print("\n" + "="*60)
        print("PREPROCESSING SELESAI")
        print("="*60)
        
        return self.df


if __name__ == "__main__":
    # Path file input dan output
    input_file = "customer_aggregated.csv"
    output_file = "customer_aggregated_preprocessed.csv"
    
    # Buat instance preprocessor
    preprocessor = CustomerDataPreprocessor(input_file)
    
    # Jalankan preprocessing
    # Set remove_outliers=False jika tidak ingin menghapus outlier
    df_preprocessed = preprocessor.run_full_preprocessing(
        output_filepath=output_file,
        remove_outliers=True  # Ubah ke False jika ingin mempertahankan semua data
    )
    
    print("\n✓ Preprocessing selesai!")
    print(f"✓ File output: {output_file}")
    print(f"✓ CustomerID sekarang bertipe: {df_preprocessed['CustomerID'].dtype}")
