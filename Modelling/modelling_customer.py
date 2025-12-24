import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CustomerModellingPipeline:
    """
    Pipeline untuk modelling High Value Customer (HVC) Classification
    
    Metodologi Supervised Learning:
    1. Load data preprocessed
    2. Create target variable: HVC berdasarkan persentil ke-75 TotalSpending
    3. Feature selection (tanpa data leakage): TotalTransaction, TotalQuantity, AvgPrice, Recency
    4. Split data (80%-20%) dengan stratified sampling
    5. Preprocessing: StandardScaler hanya untuk Logistic Regression
    6. Training: Logistic Regression (scaled) & Random Forest (unscaled)
    7. Evaluasi: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
    8. Visualisasi: Confusion Matrix, Metrics Comparison, Feature Importance
    """
    
    def __init__(self, filepath):
        """Inisialisasi dengan membaca data"""
        print("="*70)
        print("CUSTOMER CLASSIFICATION MODELLING PIPELINE")
        print("="*70)
        self.df = pd.read_csv(filepath)
        print(f"\n✓ Data berhasil dimuat: {self.df.shape[0]} baris, {self.df.shape[1]} kolom")
        
        # Initialize variables
        self.X_train_scaled = None  # For Logistic Regression
        self.X_test_scaled = None   # For Logistic Regression
        self.X_train = None         # For Random Forest (unscaled)
        self.X_test = None          # For Random Forest (unscaled)
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model_lr = None
        self.model_rf = None
        self.feature_cols = None
        self.percentile_75 = None   # Threshold untuk HVC
        
    def explore_data(self):
        """Eksplorasi data awal"""
        print("\n" + "="*70)
        print("EKSPLORASI DATA")
        print("="*70)
        
        print("\n📊 Informasi Dataset:")
        print(f"   Jumlah Customer: {len(self.df)}")
        print(f"   Kolom: {list(self.df.columns)}")
        print(f"\n   Tipe Data:")
        for col in self.df.columns:
            print(f"   - {col}: {self.df[col].dtype}")
        
        print("\n📈 Statistik Deskriptif:")
        print(self.df.describe())
        
        print("\n🔍 Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("   ✓ Tidak ada missing values")
            
        print("\n📋 Sample Data (5 baris pertama):")
        print(self.df.head())
        
    def create_hvc_labels(self):
        """
        Create High Value Customer (HVC) labels berdasarkan persentil ke-75
        
        Metodologi:
        - HVC = 1 jika TotalSpending > persentil ke-75
        - Non-HVC = 0 jika TotalSpending ≤ persentil ke-75
        - TotalSpending TIDAK digunakan sebagai fitur (menghindari data leakage)
        """
        print("\n" + "="*70)
        print("MEMBUAT LABEL TARGET: HIGH VALUE CUSTOMER (HVC)")
        print("="*70)
        
        # Statistik TotalSpending
        spending_min = self.df['TotalSpending'].min()
        spending_max = self.df['TotalSpending'].max()
        spending_mean = self.df['TotalSpending'].mean()
        spending_median = self.df['TotalSpending'].median()
        
        print(f"\n📊 Statistik TotalSpending:")
        print(f"   Jumlah Customer: {len(self.df)}")
        print(f"   Minimum: £{spending_min:.2f}")
        print(f"   Maximum: £{spending_max:.2f}")
        print(f"   Mean: £{spending_mean:.2f}")
        print(f"   Median: £{spending_median:.2f}")
        
        # Hitung persentil ke-75 (data-driven threshold)
        self.percentile_75 = self.df['TotalSpending'].quantile(0.75)
        
        print(f"\n🎯 Threshold HVC (Data-Driven):")
        print(f"   Persentil ke-75: £{self.percentile_75:.2f}")
        print(f"   HVC (1): TotalSpending > £{self.percentile_75:.2f}")
        print(f"   Non-HVC (0): TotalSpending ≤ £{self.percentile_75:.2f}")
        
        # Create binary labels
        self.df['HVC'] = (self.df['TotalSpending'] > self.percentile_75).astype(int)
        
        # Show distribution
        print(f"\n📈 Distribusi Label HVC:")
        hvc_counts = self.df['HVC'].value_counts().sort_index()
        
        for label in [0, 1]:
            count = hvc_counts[label]
            label_name = 'Non-HVC' if label == 0 else 'HVC'
            segment_data = self.df[self.df['HVC'] == label]['TotalSpending']
            avg_spending = segment_data.mean()
            min_spending = segment_data.min()
            max_spending = segment_data.max()
            print(f"\n   {label_name} ({label}): {count:4d} customers ({count/len(self.df)*100:.1f}%)")
            print(f"      Avg Spending: £{avg_spending:.2f}")
            print(f"      Min Spending: £{min_spending:.2f}")
            print(f"      Max Spending: £{max_spending:.2f}")
        
        print(f"\n✓ Label HVC berhasil dibuat!")
        print(f"✓ Total data untuk modelling: {len(self.df)} customers")
        print(f"\n⚠️  CATATAN: TotalSpending TIDAK akan digunakan sebagai fitur (menghindari data leakage)")
        
        return self.df
        
    def preprocessing_and_split(self, test_size=0.2, random_state=42):
        """
        Preprocessing dan split data (metodologi tanpa data leakage)
        
        Steps:
        1. Pilih 4 fitur: TotalTransaction, TotalQuantity, AvgPrice, Recency
        2. TIDAK gunakan TotalSpending (untuk menghindari data leakage)
        3. Split data 80%-20% dengan stratified sampling
        4. StandardScaler hanya untuk Logistic Regression
        5. Random Forest gunakan data unscaled (original)
        """
        print("\n" + "="*70)
        print("PREPROCESSING & SPLITTING DATA")
        print("="*70)
        
        # Simpan CustomerID
        customer_ids = self.df['CustomerID'].copy()
        
        # Target variable
        y = self.df['HVC'].copy()
        
        # Pilih HANYA 4 fitur (menghindari data leakage)
        self.feature_cols = ['TotalTransaction', 'TotalQuantity', 'AvgPrice', 'Recency']
        X = self.df[self.feature_cols].copy()
        
        print(f"\n📌 Fitur yang digunakan (tanpa data leakage):")
        for i, col in enumerate(self.feature_cols, 1):
            print(f"   {i}. {col}")
        
        print(f"\n🚫 Fitur yang TIDAK digunakan:")
        print(f"   - CustomerID (bukan fitur prediktif)")
        print(f"   - TotalSpending (digunakan untuk membuat label, akan menyebabkan data leakage)")
        
        print(f"\n🎯 Target Variable: HVC (High Value Customer)")
        print(f"   Classes: {sorted(y.unique())} (0=Non-HVC, 1=HVC)")
        
        # Split data DULU sebelum scaling (best practice)
        print(f"\n✂️  Splitting data: Train ({int((1-test_size)*100)}%) - Test ({int(test_size*100)}%)")
        
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, customer_ids,
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class proportion
        )
        
        # Reset index
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        ids_train = ids_train.reset_index(drop=True)
        ids_test = ids_test.reset_index(drop=True)
        
        print(f"\n✓ Split selesai!")
        print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Feature Scaling - HANYA untuk Logistic Regression
        print(f"\n⚙️  Feature Scaling:")
        print(f"   - Logistic Regression: Menggunakan StandardScaler (SCALED)")
        print(f"   - Random Forest: Menggunakan data original (UNSCALED)")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)  # Fit on train only
        X_test_scaled = self.scaler.transform(X_test)        # Transform test
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_cols)
        
        # Store both versions
        self.X_train_scaled = X_train_scaled  # For Logistic Regression
        self.X_test_scaled = X_test_scaled    # For Logistic Regression
        self.X_train = X_train                # For Random Forest (unscaled)
        self.X_test = X_test                  # For Random Forest (unscaled)
        self.y_train = y_train
        self.y_test = y_test
        self.customer_ids_train = ids_train
        self.customer_ids_test = ids_test
        
        # Show class distribution
        print(f"\n📊 Distribusi Class (Stratified):")
        print("\n   Training Set:")
        train_dist = y_train.value_counts().sort_index()
        for cls, count in train_dist.items():
            label_name = 'Non-HVC' if cls == 0 else 'HVC'
            print(f"      {label_name} ({cls}): {count:4d} ({count/len(y_train)*100:.1f}%)")
        
        print("\n   Test Set:")
        test_dist = y_test.value_counts().sort_index()
        for cls, count in test_dist.items():
            label_name = 'Non-HVC' if cls == 0 else 'HVC'
            print(f"      {label_name} ({cls}): {count:4d} ({count/len(y_test)*100:.1f}%)")
        
    def review_preprocessing(self):
        """Review hasil preprocessing sebelum modelling"""
        print("\n" + "="*70)
        print("REVIEW HASIL PREPROCESSING")
        print("="*70)
        
        print("\n📊 DATA UNTUK LOGISTIC REGRESSION (SCALED)")
        print("-" * 70)
        print(f"Training Shape: {self.X_train_scaled.shape}")
        print(f"Test Shape: {self.X_test_scaled.shape}")
        print("\nStatistik Features (Scaled) - Training Set:")
        print(self.X_train_scaled.describe())
        print("\nSample Data (Scaled) - 5 baris pertama:")
        print(self.X_train_scaled.head())
        
        print("\n\n📊 DATA UNTUK RANDOM FOREST (UNSCALED)")
        print("-" * 70)
        print(f"Training Shape: {self.X_train.shape}")
        print(f"Test Shape: {self.X_test.shape}")
        print("\nStatistik Features (Unscaled) - Training Set:")
        print(self.X_train.describe())
        print("\nSample Data (Unscaled) - 5 baris pertama:")
        print(self.X_train.head())
        
        print("\n\n🎯 TARGET VARIABLE")
        print("-" * 70)
        print(f"Shape: {self.y_train.shape}")
        print("\nSample Labels - 10 baris pertama:")
        print(self.y_train.head(10))
        print(f"\nDistribusi:")
        for cls, count in self.y_train.value_counts().sort_index().items():
            label_name = 'Non-HVC' if cls == 0 else 'HVC'
            print(f"   {label_name} ({cls}): {count} ({count/len(self.y_train)*100:.1f}%)")
        
        # Verifikasi scaling
        print("\n\n✅ VERIFIKASI SCALING")
        print("-" * 70)
        print("Scaled data (untuk LR) - Mean harus ~0, Std ~1:")
        for col in self.feature_cols:
            print(f"   {col:20s}: Mean={self.X_train_scaled[col].mean():.4f}, Std={self.X_train_scaled[col].std():.4f}")
        
        print("\nUnscaled data (untuk RF) - Nilai original:")
        for col in self.feature_cols:
            print(f"   {col:20s}: Mean={self.X_train[col].mean():.2f}, Std={self.X_train[col].std():.2f}")
        
    def confirm_continue(self):
        """Minta konfirmasi user untuk melanjutkan modelling"""
        print("\n" + "="*70)
        print("KONFIRMASI")
        print("="*70)
        print("\n⚠️  Data sudah dipreprocess dan di-split.")
        print("   Apakah Anda ingin melanjutkan ke tahap MODELLING?")
        print("\n   Modelling yang akan dilakukan:")
        print("   1. Logistic Regression - High Value Customer (HVC) Classification")
        print("      → Menggunakan SCALED data")
        print("   2. Random Forest Classifier - HVC Classification")
        print("      → Menggunakan UNSCALED data")
        print("\n   Evaluasi:")
        print("   - Accuracy, Precision, Recall, F1-Score")
        print("   - Confusion Matrix")
        print("   - Classification Report")
        print("   - Feature Importance (Random Forest)")
        
        while True:
            response = input("\n👉 Lanjutkan? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                print("\n✓ Melanjutkan ke tahap modelling...\n")
                return True
            elif response in ['n', 'no']:
                print("\n✗ Modelling dibatalkan oleh user.")
                return False
            else:
                print("   ⚠️  Input tidak valid. Silakan masukkan 'y' atau 'n'")
    
    def train_logistic_regression(self):
        """
        Training Logistic Regression dengan SCALED data
        """
        print("\n" + "="*70)
        print("TRAINING LOGISTIC REGRESSION (SCALED DATA)")
        print("="*70)
        
        print(f"\n⚙️  Training Logistic Regression...")
        print(f"   - Data: SCALED (StandardScaler)")
        print(f"   - Binary Classification (0=Non-HVC, 1=HVC)")
        print(f"   - max_iter: 1000")
        print(f"   - solver: default (lbfgs untuk binary)")
        
        self.model_lr = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        self.model_lr.fit(self.X_train_scaled, self.y_train)
        
        # Prediksi
        y_train_pred_lr = self.model_lr.predict(self.X_train_scaled)
        y_test_pred_lr = self.model_lr.predict(self.X_test_scaled)
        
        print(f"\n✓ Training selesai!")
        print(f"   Training accuracy: {accuracy_score(self.y_train, y_train_pred_lr):.4f}")
        print(f"   Test accuracy: {accuracy_score(self.y_test, y_test_pred_lr):.4f}")
        
        # Store predictions
        self.y_train_pred_lr = y_train_pred_lr
        self.y_test_pred_lr = y_test_pred_lr
        
        return y_train_pred_lr, y_test_pred_lr
    
    def train_random_forest(self, n_estimators=100):
        """
        Training Random Forest dengan UNSCALED data
        """
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST (UNSCALED DATA)")
        print("="*70)
        
        print(f"\n⚙️  Training Random Forest dengan {n_estimators} trees...")
        print(f"   - Data: UNSCALED (original values)")
        print(f"   - Binary Classification (0=Non-HVC, 1=HVC)")
        print(f"   - n_estimators: {n_estimators}")
        
        self.model_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.model_rf.fit(self.X_train, self.y_train)
        
        # Prediksi
        y_train_pred_rf = self.model_rf.predict(self.X_train)
        y_test_pred_rf = self.model_rf.predict(self.X_test)
        
        print(f"\n✓ Training selesai!")
        print(f"   Training accuracy: {accuracy_score(self.y_train, y_train_pred_rf):.4f}")
        print(f"   Test accuracy: {accuracy_score(self.y_test, y_test_pred_rf):.4f}")
        
        # Store predictions
        self.y_train_pred_rf = y_train_pred_rf
        self.y_test_pred_rf = y_test_pred_rf
        
        # Feature importance dengan interpretasi
        print(f"\n📊 FEATURE IMPORTANCE:")
        feature_importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': self.model_rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n   Ranking fitur paling berpengaruh untuk prediksi HVC:")
        for idx, row in feature_importance.iterrows():
            print(f"   {idx+1}. {row['Feature']:20s}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")
        
        # Interpretasi
        top_feature = feature_importance.iloc[0]
        print(f"\n💡 Interpretasi:")
        print(f"   Fitur paling berpengaruh: {top_feature['Feature']}")
        print(f"   Importance: {top_feature['Importance']:.4f} ({top_feature['Importance']*100:.1f}%)")
        
        return y_train_pred_rf, y_test_pred_rf
    
    def evaluate_model(self, model_name, y_train_true, y_train_pred, y_test_true, y_test_pred):
        """
        Evaluasi model dengan berbagai metrik
        """
        print("\n" + "="*70)
        print(f"EVALUASI MODEL: {model_name}")
        print("="*70)
        
        # Training metrics
        train_acc = accuracy_score(y_train_true, y_train_pred)
        train_precision = precision_score(y_train_true, y_train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train_true, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train_true, y_train_pred, average='weighted', zero_division=0)
        
        # Test metrics
        test_acc = accuracy_score(y_test_true, y_test_pred)
        test_precision = precision_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
        
        print("\n📈 METRIK EVALUASI")
        print("-" * 70)
        print("\nTraining Set:")
        print(f"   Accuracy:  {train_acc:.4f}")
        print(f"   Precision: {train_precision:.4f}")
        print(f"   Recall:    {train_recall:.4f}")
        print(f"   F1-Score:  {train_f1:.4f}")
        
        print("\nTest Set:")
        print(f"   Accuracy:  {test_acc:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall:    {test_recall:.4f}")
        print(f"   F1-Score:  {test_f1:.4f}")
        
        # Classification Report
        print("\n📋 CLASSIFICATION REPORT (Test Set):")
        print("-" * 70)
        print(classification_report(y_test_true, y_test_pred, zero_division=0))
        
        # Confusion Matrix
        print("\n📊 CONFUSION MATRIX (Test Set):")
        print("-" * 70)
        cm = confusion_matrix(y_test_true, y_test_pred)
        print(cm)
        
        return {
            'train_acc': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': cm
        }
    
    def compare_models(self):
        """
        Membandingkan performa Logistic Regression vs Random Forest
        """
        print("\n" + "="*70)
        print("PERBANDINGAN MODEL")
        print("="*70)
        
        # Evaluate both models
        print("\n" + "="*35 + " 1/2 " + "="*35)
        metrics_lr = self.evaluate_model(
            "Logistic Regression",
            self.y_train, self.y_train_pred_lr,
            self.y_test, self.y_test_pred_lr
        )
        
        print("\n" + "="*35 + " 2/2 " + "="*35)
        metrics_rf = self.evaluate_model(
            "Random Forest",
            self.y_train, self.y_train_pred_rf,
            self.y_test, self.y_test_pred_rf
        )
        
        # Comparison table
        print("\n" + "="*70)
        print("TABEL PERBANDINGAN (Test Set)")
        print("="*70)
        
        comparison = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Logistic Regression': [
                metrics_lr['test_acc'],
                metrics_lr['test_precision'],
                metrics_lr['test_recall'],
                metrics_lr['test_f1']
            ],
            'Random Forest': [
                metrics_rf['test_acc'],
                metrics_rf['test_precision'],
                metrics_rf['test_recall'],
                metrics_rf['test_f1']
            ]
        })
        
        # Determine winner for each metric
        comparison['Winner'] = comparison.apply(
            lambda row: 'Random Forest' if row['Random Forest'] > row['Logistic Regression'] 
            else ('Logistic Regression' if row['Logistic Regression'] > row['Random Forest'] else 'Tie'),
            axis=1
        )
        
        print("\n", comparison.to_string(index=False))
        
        # Overall winner
        lr_wins = (comparison['Winner'] == 'Logistic Regression').sum()
        rf_wins = (comparison['Winner'] == 'Random Forest').sum()
        
        print("\n💡 KESIMPULAN:")
        if rf_wins > lr_wins:
            print(f"   🏆 Random Forest menang di {rf_wins} metrik")
            print(f"   → Random Forest adalah model terbaik untuk dataset ini")
        elif lr_wins > rf_wins:
            print(f"   🏆 Logistic Regression menang di {lr_wins} metrik")
            print(f"   → Logistic Regression adalah model terbaik untuk dataset ini")
        else:
            print(f"   🤝 Kedua model memiliki performa yang sebanding")
        
        return metrics_lr, metrics_rf, comparison
    
    def visualize_results(self, metrics_lr, metrics_rf):
        """
        Visualisasi hasil modelling
        """
        print("\n" + "="*70)
        print("VISUALISASI HASIL")
        print("="*70)
        
        # 1. Confusion Matrices
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Logistic Regression CM
        cm_lr = metrics_lr['confusion_matrix']
        labels = sorted(self.y_test.unique())
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('Actual', fontsize=12)
        
        # Random Forest CM
        cm_rf = metrics_rf['confusion_matrix']
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
                   xticklabels=labels, yticklabels=labels, ax=axes[1])
        axes[1].set_title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrices disimpan sebagai: confusion_matrices.png")
        plt.show()
        
        # 2. Metrics Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        lr_scores = [metrics_lr['test_acc'], metrics_lr['test_precision'], 
                    metrics_lr['test_recall'], metrics_lr['test_f1']]
        rf_scores = [metrics_rf['test_acc'], metrics_rf['test_precision'],
                    metrics_rf['test_recall'], metrics_rf['test_f1']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='#3498db')
        bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='#2ecc71')
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison (Test Set)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Metrics comparison disimpan sebagai: metrics_comparison.png")
        plt.show()
        
        # 3. Feature Importance (Random Forest only)
        if self.model_rf is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            feature_importance = pd.DataFrame({
                'Feature': self.feature_cols,
                'Importance': self.model_rf.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='#e74c3c')
            ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
            ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("✓ Feature importance disimpan sebagai: feature_importance.png")
            plt.show()
        
        print("\n✓ Visualisasi selesai!")
    
    def save_results(self, output_file='hvc_predictions.csv'):
        """
        Simpan hasil classification HVC
        Format: CustomerID, ActualLabel, LogisticRegressionPrediction, RandomForestPrediction
        """
        print("\n" + "="*70)
        print("MENYIMPAN HASIL PREDIKSI")
        print("="*70)
        
        # Gabungkan semua data (train + test)
        # Training data
        train_result = pd.DataFrame({
            'CustomerID': self.customer_ids_train,
            'ActualLabel': self.y_train,
            'LogisticRegressionPrediction': self.y_train_pred_lr,
            'RandomForestPrediction': self.y_train_pred_rf,
            'DataSet': 'Train'
        })
        
        # Test data
        test_result = pd.DataFrame({
            'CustomerID': self.customer_ids_test,
            'ActualLabel': self.y_test,
            'LogisticRegressionPrediction': self.y_test_pred_lr,
            'RandomForestPrediction': self.y_test_pred_rf,
            'DataSet': 'Test'
        })
        
        # Gabungkan
        result_df = pd.concat([train_result, test_result], ignore_index=True)
        
        # Tambahkan label description
        result_df['ActualLabel_Desc'] = result_df['ActualLabel'].map({0: 'Non-HVC', 1: 'HVC'})
        result_df['LR_Prediction_Desc'] = result_df['LogisticRegressionPrediction'].map({0: 'Non-HVC', 1: 'HVC'})
        result_df['RF_Prediction_Desc'] = result_df['RandomForestPrediction'].map({0: 'Non-HVC', 1: 'HVC'})
        
        # Reorder columns
        cols = ['CustomerID', 'ActualLabel', 'ActualLabel_Desc', 
                'LogisticRegressionPrediction', 'LR_Prediction_Desc',
                'RandomForestPrediction', 'RF_Prediction_Desc', 'DataSet']
        result_df = result_df[cols]
        
        # Save
        result_df.to_csv(output_file, index=False)
        print(f"\n✓ Hasil prediksi HVC disimpan ke: {output_file}")
        print(f"   Total records: {len(result_df)}")
        print(f"   - Training: {len(train_result)} ({len(train_result)/len(result_df)*100:.1f}%)")
        print(f"   - Test: {len(test_result)} ({len(test_result)/len(result_df)*100:.1f}%)")
        print(f"\n   Columns:")
        for col in cols:
            print(f"      - {col}")
        
        # Summary statistics
        print(f"\n📊 Summary Prediksi (Test Set):")
        test_lr_acc = accuracy_score(test_result['ActualLabel'], test_result['LogisticRegressionPrediction'])
        test_rf_acc = accuracy_score(test_result['ActualLabel'], test_result['RandomForestPrediction'])
        print(f"   Logistic Regression Accuracy: {test_lr_acc:.4f}")
        print(f"   Random Forest Accuracy: {test_rf_acc:.4f}")
        
        return result_df
    
    def run_full_pipeline(self, test_size=0.2, save_output=True):
        """
        Menjalankan seluruh pipeline classification
        """
        try:
            # 1. Eksplorasi data
            self.explore_data()
            
            # 2. Create HVC Labels (data-driven dengan persentil ke-75)
            self.create_hvc_labels()
            
            # 3. Preprocessing & Split (4 fitur tanpa data leakage)
            self.preprocessing_and_split(test_size=test_size)
            
            # 4. Review preprocessing
            self.review_preprocessing()
            
            # 5. Konfirmasi user
            if not self.confirm_continue():
                print("\n❌ Pipeline dihentikan.")
                return None
            
            # 6. Training Logistic Regression
            self.train_logistic_regression()
            
            # 7. Training Random Forest
            self.train_random_forest(n_estimators=100)
            
            # 8. Compare models
            metrics_lr, metrics_rf, comparison = self.compare_models()
            
            # 9. Visualisasi
            self.visualize_results(metrics_lr, metrics_rf)
            
            # 10. Save results
            if save_output:
                result_df = self.save_results()
            
            print("\n" + "="*70)
            print("✅ PIPELINE HVC CLASSIFICATION SELESAI!")
            print("="*70)
            print("\n📁 File yang dihasilkan:")
            print("   1. confusion_matrices.png - Confusion matrices kedua model")
            print("   2. metrics_comparison.png - Perbandingan metrik (Accuracy, Precision, Recall, F1)")
            print("   3. feature_importance.png - Feature importance Random Forest")
            if save_output:
                print("   4. hvc_predictions.csv - Hasil prediksi HVC kedua model")
            print("\n💡 Model terbaik untuk prediksi High Value Customer sudah ditentukan berdasarkan evaluasi metrik.")
            
            return result_df if save_output else comparison
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Path file input
    input_file = "customer_aggregated_preprocessed.csv"
    
    # Buat instance pipeline
    pipeline = CustomerModellingPipeline(input_file)
    
    # Jalankan full pipeline High Value Customer Classification
    # Metodologi:
    # - Create HVC labels (data-driven, persentil ke-75)
    # - Feature selection (4 fitur tanpa data leakage)
    # - Split 80%-20% (stratified)
    # - Preprocessing: Scaled untuk LR, Unscaled untuk RF
    # - Training: Logistic Regression & Random Forest
    # - Evaluasi: Accuracy, Precision, Recall, F1-Score
    # - Visualisasi & Save hasil
    result = pipeline.run_full_pipeline(
        test_size=0.2,  # 80% train, 20% test
        save_output=True
    )
    
    print("\n" + "="*70)
    print("🎉 HIGH VALUE CUSTOMER CLASSIFICATION SELESAI!")
    print("="*70)
