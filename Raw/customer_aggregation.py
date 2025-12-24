import pandas as pd
import numpy as np
from datetime import datetime

class CustomerAggregator:
    """
    Class untuk agregasi data transaksi menjadi level pelanggan
    
    Fitur-fitur agregasi yang dibuat:
    1. TotalSpending - Total nilai belanja pelanggan
    2. TotalTransaction - Jumlah transaksi unik
    3. TotalQuantity - Total unit barang yang dibeli
    4. AvgPrice - Rata-rata harga produk yang dibeli
    5. Recency - Selisih hari antara transaksi terakhir dengan tanggal acuan
    """
    
    def __init__(self, filepath):
        """Inisialisasi dengan membaca file CSV yang sudah dipreprocess"""
        print("="*60)
        print("MEMUAT DATA PREPROCESSED")
        print("="*60)
        self.df = pd.read_csv(filepath)
        
        # Konversi InvoiceDate ke datetime jika belum
        if self.df['InvoiceDate'].dtype == 'object':
            self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        
        print(f"Data berhasil dimuat: {self.df.shape[0]:,} baris, {self.df.shape[1]} kolom")
        print(f"Jumlah pelanggan unik: {self.df['Customer ID'].nunique():,}")
        print(f"Periode data: {self.df['InvoiceDate'].min()} s/d {self.df['InvoiceDate'].max()}")
        
    def aggregate_customer_features(self):
        """
        Agregasi data ke level pelanggan
        Membuat fitur-fitur perilaku pelanggan
        """
        print("\n" + "="*60)
        print("AGREGASI DATA KE LEVEL PELANGGAN")
        print("="*60)
        
        # Tanggal acuan untuk menghitung Recency (tanggal terakhir dalam dataset + 1 hari)
        reference_date = self.df['InvoiceDate'].max() + pd.Timedelta(days=1)
        print(f"\nTanggal acuan untuk Recency: {reference_date}")
        
        # Agregasi per Customer ID
        print("\nMelakukan agregasi...")
        
        customer_agg = self.df.groupby('Customer ID').agg({
            # TotalSpending: Total nilai belanja
            'TotalAmount': 'sum',
            
            # TotalTransaction: Jumlah transaksi unik (invoice unik)
            'Invoice': 'nunique',
            
            # TotalQuantity: Total unit barang yang dibeli
            'Quantity': 'sum',
            
            # AvgPrice: Rata-rata harga produk
            'Price': 'mean',
            
            # Untuk menghitung Recency: tanggal transaksi terakhir
            'InvoiceDate': 'max'
        }).reset_index()
        
        # Rename kolom agar lebih deskriptif
        customer_agg.columns = ['CustomerID', 'TotalSpending', 'TotalTransaction', 
                                'TotalQuantity', 'AvgPrice', 'LastTransactionDate']
        
        # Hitung Recency (selisih hari dari transaksi terakhir ke tanggal acuan)
        customer_agg['Recency'] = (reference_date - customer_agg['LastTransactionDate']).dt.days
        
        # Bulatkan nilai desimal untuk AvgPrice
        customer_agg['AvgPrice'] = customer_agg['AvgPrice'].round(2)
        customer_agg['TotalSpending'] = customer_agg['TotalSpending'].round(2)
        
        print("\n✓ Agregasi selesai!")
        
        self.customer_df = customer_agg
        return customer_agg
    
    def show_aggregation_summary(self):
        """Menampilkan ringkasan hasil agregasi"""
        print("\n" + "="*60)
        print("RINGKASAN HASIL AGREGASI")
        print("="*60)
        
        print(f"\nJumlah pelanggan: {len(self.customer_df):,}")
        
        print("\n" + "-"*60)
        print("STATISTIK FITUR AGREGASI")
        print("-"*60)
        
        # TotalSpending
        print(f"\nTotalSpending:")
        print(f"  • Min: ${self.customer_df['TotalSpending'].min():,.2f}")
        print(f"  • Max: ${self.customer_df['TotalSpending'].max():,.2f}")
        print(f"  • Mean: ${self.customer_df['TotalSpending'].mean():,.2f}")
        print(f"  • Median: ${self.customer_df['TotalSpending'].median():,.2f}")
        
        # TotalTransaction
        print(f"\nTotalTransaction:")
        print(f"  • Min: {self.customer_df['TotalTransaction'].min()}")
        print(f"  • Max: {self.customer_df['TotalTransaction'].max()}")
        print(f"  • Mean: {self.customer_df['TotalTransaction'].mean():.2f}")
        print(f"  • Median: {self.customer_df['TotalTransaction'].median():.0f}")
        
        # TotalQuantity
        print(f"\nTotalQuantity:")
        print(f"  • Min: {self.customer_df['TotalQuantity'].min():,}")
        print(f"  • Max: {self.customer_df['TotalQuantity'].max():,}")
        print(f"  • Mean: {self.customer_df['TotalQuantity'].mean():,.2f}")
        print(f"  • Median: {self.customer_df['TotalQuantity'].median():,.0f}")
        
        # AvgPrice
        print(f"\nAvgPrice:")
        print(f"  • Min: ${self.customer_df['AvgPrice'].min():.2f}")
        print(f"  • Max: ${self.customer_df['AvgPrice'].max():.2f}")
        print(f"  • Mean: ${self.customer_df['AvgPrice'].mean():.2f}")
        print(f"  • Median: ${self.customer_df['AvgPrice'].median():.2f}")
        
        # Recency
        print(f"\nRecency (hari sejak transaksi terakhir):")
        print(f"  • Min: {self.customer_df['Recency'].min()} hari")
        print(f"  • Max: {self.customer_df['Recency'].max()} hari")
        print(f"  • Mean: {self.customer_df['Recency'].mean():.2f} hari")
        print(f"  • Median: {self.customer_df['Recency'].median():.0f} hari")
        
        # Top 10 pelanggan berdasarkan TotalSpending
        print("\n" + "-"*60)
        print("TOP 10 PELANGGAN BERDASARKAN TOTAL SPENDING")
        print("-"*60)
        top_customers = self.customer_df.nlargest(10, 'TotalSpending')[
            ['CustomerID', 'TotalSpending', 'TotalTransaction', 'TotalQuantity', 'Recency']
        ]
        print(top_customers.to_string(index=False))
        
    def show_data_info(self):
        """Menampilkan informasi dataset agregasi"""
        print("\n" + "="*60)
        print("INFORMASI DATASET AGREGASI")
        print("="*60)
        
        print("\nTipe data:")
        print(self.customer_df.dtypes)
        
        print("\nStatistik deskriptif:")
        print(self.customer_df.describe())
        
        print("\nMissing values:")
        print(self.customer_df.isnull().sum())
        
        print("\nSample data (5 baris pertama):")
        print(self.customer_df.head())
        
    def save_aggregated_data(self, output_path):
        """Simpan data agregasi ke CSV"""
        # Drop kolom LastTransactionDate sebelum menyimpan (hanya untuk referensi)
        df_to_save = self.customer_df.drop('LastTransactionDate', axis=1)
        
        df_to_save.to_csv(output_path, index=False)
        print("\n" + "="*60)
        print("MENYIMPAN DATA")
        print("="*60)
        print(f"✓ Data agregasi berhasil disimpan ke: {output_path}")
        print(f"✓ Total pelanggan: {len(df_to_save):,}")
        print(f"✓ Total kolom: {df_to_save.shape[1]}")
        print(f"\nKolom-kolom dalam dataset agregasi:")
        for col in df_to_save.columns:
            print(f"  • {col}")
        
    def run_full_aggregation(self, output_path='customer_aggregated.csv'):
        """Jalankan semua tahap agregasi"""
        self.aggregate_customer_features()
        self.show_aggregation_summary()
        self.show_data_info()
        self.save_aggregated_data(output_path)
        
        return self.customer_df


# ============================================================================
# CARA PENGGUNAAN
# ============================================================================

if __name__ == "__main__":
    # Path file input (hasil preprocessing)
    input_file = "online_retail_II_preprocessed.csv"
    output_file = "customer_aggregated.csv"
    
    # Inisialisasi aggregator
    aggregator = CustomerAggregator(input_file)
    
    # Jalankan agregasi lengkap
    customer_df = aggregator.run_full_aggregation(output_file)
    
    print("\n" + "="*60)
    print("AGREGASI SELESAI!")
    print("="*60)
    print("\nFitur-fitur yang dibuat:")
    print("  1. TotalSpending     - Total nilai belanja pelanggan")
    print("  2. TotalTransaction  - Jumlah transaksi unik")
    print("  3. TotalQuantity     - Total unit barang yang dibeli")
    print("  4. AvgPrice          - Rata-rata harga produk yang dibeli")
    print("  5. Recency           - Selisih hari dari transaksi terakhir")
    print("\nDataset siap untuk pemodelan supervised learning!")
