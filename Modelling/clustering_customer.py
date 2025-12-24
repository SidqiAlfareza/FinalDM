import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    """
    Pipeline untuk Customer Segmentation menggunakan K-Means Clustering
    
    Tahapan:
    1. Load & Eksplorasi Data
    2. Seleksi Fitur (5 fitur numerik)
    3. Preprocessing dengan StandardScaler
    4. Elbow Method untuk menentukan K optimal
    5. Training K-Means Clustering
    6. Analisis & Interpretasi Cluster
    7. Visualisasi dengan PCA
    8. Export hasil clustering
    """
    
    def __init__(self, filepath):
        """Inisialisasi dengan membaca data"""
        print("="*70)
        print("CUSTOMER SEGMENTATION - K-MEANS CLUSTERING")
        print("="*70)
        self.df = pd.read_csv(filepath)
        print(f"\n✓ Data berhasil dimuat: {self.df.shape[0]} baris, {self.df.shape[1]} kolom")
        
        # Initialize variables
        self.scaler = None
        self.kmeans_model = None
        self.optimal_k = None
        self.feature_cols = None
        self.X_scaled = None
        self.pca_model = None
        
    def explore_data(self):
        """Eksplorasi data awal"""
        print("\n" + "="*70)
        print("TAHAP 1: EKSPLORASI DATA")
        print("="*70)
        
        print("\n📊 Informasi Dataset:")
        print(f"   Jumlah Customer: {len(self.df)}")
        print(f"   Jumlah Kolom: {len(self.df.columns)}")
        
        print("\n📋 5 Baris Pertama:")
        print(self.df.head())
        
        print("\n📈 Statistik Deskriptif:")
        print(self.df.describe())
        
        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("   ✓ Tidak ada missing values")
        
        print("\n📊 Tipe Data:")
        print(self.df.dtypes)
        
    def select_features(self):
        """Seleksi fitur untuk clustering"""
        print("\n" + "="*70)
        print("TAHAP 2: SELEKSI FITUR")
        print("="*70)
        
        # Fitur untuk clustering (exclude CustomerID)
        self.feature_cols = ['TotalSpending', 'TotalTransaction', 'TotalQuantity', 
                            'AvgPrice', 'Recency']
        
        print(f"\n📌 Fitur yang digunakan untuk clustering:")
        for i, col in enumerate(self.feature_cols, 1):
            print(f"   {i}. {col}")
        
        print(f"\n🚫 Fitur yang TIDAK digunakan:")
        print(f"   - CustomerID (identifier, bukan fitur deskriptif)")
        
        # Verify all features exist
        missing_cols = [col for col in self.feature_cols if col not in self.df.columns]
        if missing_cols:
            print(f"\n❌ Error: Kolom tidak ditemukan: {missing_cols}")
            return False
        
        print(f"\n✓ Semua fitur tersedia untuk clustering")
        return True
        
    def preprocessing(self):
        """Preprocessing dengan StandardScaler"""
        print("\n" + "="*70)
        print("TAHAP 3: PREPROCESSING")
        print("="*70)
        
        # Extract features
        X = self.df[self.feature_cols].copy()
        
        print(f"\n📊 Data Sebelum Scaling:")
        print(X.describe())
        
        # Alasan scaling
        print(f"\n💡 ALASAN PENGGUNAAN STANDARDSCALER:")
        print(f"   1. K-Means menggunakan jarak Euclidean")
        print(f"   2. Fitur dengan skala besar akan mendominasi perhitungan jarak")
        print(f"   3. StandardScaler mengubah semua fitur ke mean=0, std=1")
        print(f"   4. Memastikan setiap fitur berkontribusi setara dalam clustering")
        
        # Apply StandardScaler
        print(f"\n⚙️  Melakukan StandardScaler...")
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.X_scaled_df = pd.DataFrame(self.X_scaled, columns=self.feature_cols)
        
        print(f"\n📊 Data Setelah Scaling:")
        print(self.X_scaled_df.describe())
        
        print(f"\n✓ Preprocessing selesai!")
        print(f"   Data sudah di-scaling dengan StandardScaler")
        print(f"   Shape: {self.X_scaled.shape}")
        
    def elbow_method(self, k_range=(2, 7)):
        """Elbow Method untuk menentukan K optimal"""
        print("\n" + "="*70)
        print("TAHAP 4: ELBOW METHOD - PENENTUAN K OPTIMAL")
        print("="*70)
        
        k_min, k_max = k_range
        k_values = range(k_min, k_max)
        inertias = []
        
        print(f"\n⚙️  Menghitung inertia untuk K = {k_min} sampai {k_max-1}...")
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            print(f"   K={k}: Inertia={kmeans.inertia_:.2f}")
        
        # Plot Elbow
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Jumlah Cluster (K)', fontsize=12, fontweight='bold')
        plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
        plt.title('Elbow Method untuk Menentukan K Optimal', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        
        # Annotate points
        for k, inertia in zip(k_values, inertias):
            plt.annotate(f'{inertia:.0f}', 
                        xy=(k, inertia), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=9)
        
        plt.tight_layout()
        plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Grafik Elbow Method disimpan sebagai: elbow_method.png")
        plt.show()
        
        # Analisis
        print(f"\n📊 ANALISIS ELBOW METHOD:")
        print(f"\n   Penurunan Inertia:")
        for i in range(len(inertias)-1):
            k = k_values[i]
            decrease = inertias[i] - inertias[i+1]
            decrease_pct = (decrease / inertias[i]) * 100
            print(f"   K={k} → K={k+1}: Δ{decrease:.2f} ({decrease_pct:.1f}% penurunan)")
        
        # Determine optimal K (manual analysis)
        print(f"\n💡 PENENTUAN K OPTIMAL:")
        print(f"   - Cari 'elbow' (siku) pada grafik")
        print(f"   - Titik dimana penurunan inertia mulai melambat")
        print(f"   - Trade-off antara kompleksitas dan kualitas clustering")
        
        # Auto-suggest (simple heuristic)
        decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        decrease_ratios = [decreases[i] / decreases[i+1] if i < len(decreases)-1 else 0 
                          for i in range(len(decreases))]
        
        # Find where decrease ratio is maximum (biggest drop change)
        suggested_k = k_values[decrease_ratios.index(max(decrease_ratios))] + 1
        
        print(f"\n   💡 Saran K berdasarkan analisis: K={suggested_k}")
        
        return inertias, suggested_k
    
    def train_kmeans(self, n_clusters):
        """Training K-Means dengan K yang dipilih"""
        print("\n" + "="*70)
        print("TAHAP 5: TRAINING K-MEANS CLUSTERING")
        print("="*70)
        
        self.optimal_k = n_clusters
        
        print(f"\n⚙️  Training K-Means dengan K={n_clusters}...")
        print(f"   - random_state: 42")
        print(f"   - n_init: 10 (jumlah inisialisasi)")
        
        self.kmeans_model = KMeans(
            n_clusters=n_clusters, 
            random_state=42,
            n_init=10
        )
        
        # Fit model
        self.kmeans_model.fit(self.X_scaled)
        
        # Predict clusters
        self.df['Cluster'] = self.kmeans_model.predict(self.X_scaled)
        
        print(f"\n✓ Training selesai!")
        print(f"   Inertia: {self.kmeans_model.inertia_:.2f}")
        print(f"   Iterations: {self.kmeans_model.n_iter_}")
        
        # Cluster distribution
        print(f"\n📊 Distribusi Cluster:")
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            print(f"   Cluster {cluster}: {count:4d} customers ({count/len(self.df)*100:.1f}%)")
        
        return self.kmeans_model
    
    def analyze_clusters(self):
        """Analisis karakteristik setiap cluster"""
        print("\n" + "="*70)
        print("TAHAP 6: ANALISIS & INTERPRETASI CLUSTER")
        print("="*70)
        
        # Calculate mean for each cluster
        cluster_analysis = self.df.groupby('Cluster')[self.feature_cols].mean()
        
        print(f"\n📊 KARAKTERISTIK RATA-RATA SETIAP CLUSTER:")
        print("="*70)
        print(cluster_analysis.to_string())
        
        # Interpretasi bisnis
        print(f"\n\n💡 INTERPRETASI BISNIS CLUSTER:")
        print("="*70)
        
        # Sort clusters by TotalSpending untuk interpretasi
        cluster_spending = cluster_analysis['TotalSpending'].sort_values()
        
        interpretations = {}
        cluster_names = {}
        
        # Determine cluster names based on spending
        for idx, (cluster, spending) in enumerate(cluster_spending.items()):
            if idx == 0:
                name = "Low Value Customer"
                interpretation = "Customer dengan spending, transaksi, dan quantity rendah"
            elif idx == len(cluster_spending) - 1:
                name = "High Value Customer"
                interpretation = "Customer dengan spending, transaksi, dan quantity tinggi"
            else:
                name = f"Medium Value Customer (Tier {idx})"
                interpretation = "Customer dengan nilai moderate"
            
            cluster_names[cluster] = name
            interpretations[cluster] = interpretation
        
        # Print interpretations
        for cluster in sorted(cluster_analysis.index):
            print(f"\n🏷️  CLUSTER {cluster}: {cluster_names[cluster]}")
            print(f"   Karakteristik:")
            for col in self.feature_cols:
                value = cluster_analysis.loc[cluster, col]
                print(f"      - {col:20s}: {value:,.2f}")
            print(f"   Interpretasi: {interpretations[cluster]}")
            print(f"   Jumlah Customer: {len(self.df[self.df['Cluster'] == cluster])}")
        
        # Add cluster names to dataframe
        self.df['ClusterName'] = self.df['Cluster'].map(cluster_names)
        
        # Overall insights
        print(f"\n\n🎯 INSIGHT UTAMA:")
        print("="*70)
        
        # Find most valuable cluster
        highest_spending_cluster = cluster_analysis['TotalSpending'].idxmax()
        lowest_spending_cluster = cluster_analysis['TotalSpending'].idxmin()
        
        print(f"1. Cluster dengan spending tertinggi: Cluster {highest_spending_cluster}")
        print(f"   - Rata-rata spending: £{cluster_analysis.loc[highest_spending_cluster, 'TotalSpending']:,.2f}")
        print(f"   - Jumlah customer: {len(self.df[self.df['Cluster'] == highest_spending_cluster])}")
        
        print(f"\n2. Cluster dengan spending terendah: Cluster {lowest_spending_cluster}")
        print(f"   - Rata-rata spending: £{cluster_analysis.loc[lowest_spending_cluster, 'TotalSpending']:,.2f}")
        print(f"   - Jumlah customer: {len(self.df[self.df['Cluster'] == lowest_spending_cluster])}")
        
        print(f"\n3. Rentang spending antar cluster:")
        print(f"   - Tertinggi: £{cluster_analysis['TotalSpending'].max():,.2f}")
        print(f"   - Terendah: £{cluster_analysis['TotalSpending'].min():,.2f}")
        print(f"   - Perbedaan: £{cluster_analysis['TotalSpending'].max() - cluster_analysis['TotalSpending'].min():,.2f}")
        
        return cluster_analysis, cluster_names
    
    def visualize_clusters(self):
        """Visualisasi cluster dengan PCA"""
        print("\n" + "="*70)
        print("TAHAP 7: VISUALISASI DENGAN PCA")
        print("="*70)
        
        print(f"\n⚙️  Melakukan PCA untuk reduksi dimensi...")
        print(f"   - Dari {len(self.feature_cols)} dimensi → 2 dimensi")
        print(f"   - Tujuan: Visualisasi cluster dalam 2D scatter plot")
        
        # Apply PCA
        self.pca_model = PCA(n_components=2, random_state=42)
        X_pca = self.pca_model.fit_transform(self.X_scaled)
        
        # Explained variance
        explained_var = self.pca_model.explained_variance_ratio_
        print(f"\n📊 Explained Variance Ratio:")
        print(f"   PC1: {explained_var[0]:.4f} ({explained_var[0]*100:.2f}%)")
        print(f"   PC2: {explained_var[1]:.4f} ({explained_var[1]*100:.2f}%)")
        print(f"   Total: {sum(explained_var):.4f} ({sum(explained_var)*100:.2f}%)")
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Plot each cluster
        colors = plt.cm.Set3(np.linspace(0, 1, self.optimal_k))
        
        for cluster in range(self.optimal_k):
            mask = self.df['Cluster'] == cluster
            cluster_name = self.df.loc[mask, 'ClusterName'].iloc[0]
            
            plt.scatter(
                X_pca[mask, 0], 
                X_pca[mask, 1],
                c=[colors[cluster]], 
                label=f'Cluster {cluster}: {cluster_name}',
                s=50,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Plot centroids
        centroids_pca = self.pca_model.transform(self.kmeans_model.cluster_centers_)
        plt.scatter(
            centroids_pca[:, 0], 
            centroids_pca[:, 1],
            c='red', 
            marker='X', 
            s=300, 
            edgecolors='black',
            linewidth=2,
            label='Centroids',
            zorder=5
        )
        
        plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=12, fontweight='bold')
        plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=12, fontweight='bold')
        plt.title('Customer Segmentation - K-Means Clustering (PCA Visualization)', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clustering_visualization.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualisasi cluster disimpan sebagai: clustering_visualization.png")
        plt.show()
        
        # Box plots untuk setiap fitur
        self._create_feature_boxplots()
    
    def _create_feature_boxplots(self):
        """Membuat box plots untuk setiap fitur per cluster"""
        print(f"\n📊 Membuat box plots untuk analisis lebih detail...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(self.feature_cols):
            ax = axes[idx]
            
            # Sort by cluster
            cluster_order = sorted(self.df['Cluster'].unique())
            
            # Box plot
            self.df.boxplot(column=col, by='Cluster', ax=ax)
            ax.set_title(f'{col} per Cluster', fontsize=12, fontweight='bold')
            ax.set_xlabel('Cluster', fontsize=10)
            ax.set_ylabel(col, fontsize=10)
            plt.sca(ax)
            plt.xticks(rotation=0)
        
        # Remove extra subplot
        if len(self.feature_cols) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.suptitle('Distribusi Fitur per Cluster', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        print(f"✓ Box plots disimpan sebagai: feature_distributions.png")
        plt.show()
    
    def save_results(self, output_file='customer_segmentation_results.csv'):
        """Simpan hasil clustering ke CSV"""
        print("\n" + "="*70)
        print("TAHAP 8: EXPORT HASIL CLUSTERING")
        print("="*70)
        
        # Select columns for output
        output_cols = ['CustomerID', 'Cluster', 'ClusterName'] + self.feature_cols
        result_df = self.df[output_cols].copy()
        
        # Sort by cluster and spending
        result_df = result_df.sort_values(['Cluster', 'TotalSpending'], ascending=[True, False])
        
        # Save to CSV
        result_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Hasil clustering disimpan ke: {output_file}")
        print(f"   Total records: {len(result_df)}")
        print(f"   Columns: {list(output_cols)}")
        
        # Summary
        print(f"\n📊 RINGKASAN HASIL CLUSTERING:")
        print("="*70)
        
        for cluster in sorted(result_df['Cluster'].unique()):
            cluster_data = result_df[result_df['Cluster'] == cluster]
            cluster_name = cluster_data['ClusterName'].iloc[0]
            
            print(f"\nCluster {cluster}: {cluster_name}")
            print(f"   Jumlah Customer: {len(cluster_data)}")
            print(f"   Rata-rata Spending: £{cluster_data['TotalSpending'].mean():,.2f}")
            print(f"   Rata-rata Transaksi: {cluster_data['TotalTransaction'].mean():.1f}")
            print(f"   Rata-rata Quantity: {cluster_data['TotalQuantity'].mean():.1f}")
        
        return result_df
    
    def run_full_pipeline(self, n_clusters=None):
        """
        Jalankan full pipeline clustering
        
        Parameters:
        -----------
        n_clusters : int or None
            Jumlah cluster. Jika None, akan ditentukan berdasarkan Elbow Method
        """
        try:
            # 1. Eksplorasi data
            self.explore_data()
            
            # 2. Seleksi fitur
            if not self.select_features():
                return None
            
            # 3. Preprocessing
            self.preprocessing()
            
            # 4. Elbow Method
            inertias, suggested_k = self.elbow_method()
            
            # 5. Determine K
            if n_clusters is None:
                print(f"\n📋 Menentukan jumlah cluster...")
                print(f"   Saran berdasarkan Elbow Method: K={suggested_k}")
                
                while True:
                    try:
                        user_input = input(f"\n👉 Masukkan jumlah cluster (atau tekan Enter untuk K={suggested_k}): ").strip()
                        if user_input == "":
                            n_clusters = suggested_k
                            break
                        else:
                            n_clusters = int(user_input)
                            if 2 <= n_clusters <= 10:
                                break
                            else:
                                print("   ⚠️  Masukkan nilai antara 2-10")
                    except ValueError:
                        print("   ⚠️  Input tidak valid. Masukkan angka.")
                
                print(f"\n✓ Menggunakan K={n_clusters}")
            
            # 6. Training K-Means
            self.train_kmeans(n_clusters)
            
            # 7. Analisis cluster
            self.analyze_clusters()
            
            # 8. Visualisasi
            self.visualize_clusters()
            
            # 9. Save results
            result_df = self.save_results()
            
            print("\n" + "="*70)
            print("✅ CUSTOMER SEGMENTATION SELESAI!")
            print("="*70)
            print("\n📁 File yang dihasilkan:")
            print("   1. elbow_method.png - Grafik Elbow Method")
            print("   2. clustering_visualization.png - Scatter plot PCA")
            print("   3. feature_distributions.png - Box plots per cluster")
            print("   4. customer_segmentation_results.csv - Dataset hasil clustering")
            
            return result_df
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Path file input
    input_file = "customer_aggregated_preprocessed.csv"
    
    # Buat instance pipeline
    segmentation = CustomerSegmentation(input_file)
    
    # Jalankan full pipeline
    # - Eksplorasi data
    # - Seleksi fitur
    # - Preprocessing dengan StandardScaler
    # - Elbow Method untuk menentukan K
    # - Training K-Means
    # - Analisis & interpretasi cluster
    # - Visualisasi dengan PCA
    # - Export hasil
    result = segmentation.run_full_pipeline(
        n_clusters=None  # None = akan ditentukan berdasarkan Elbow Method
    )
    
    print("\n" + "="*70)
    print("🎉 PROGRAM SELESAI!")
    print("="*70)
