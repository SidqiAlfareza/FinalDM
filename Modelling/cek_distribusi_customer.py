# ============================================================================
# ANALISIS DISTRIBUSI TARGET VARIABLE:  HVC vs Non-HVC
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assumsi:  Anda sudah memiliki dataframe dengan kolom HVC
# Jika belum, load dari hasil agregasi customer
df = pd.read_csv('../Raw/customer_aggregated.csv')  # Sesuaikan path file Anda

# Buat label HVC berdasarkan persentil ke-75
threshold_75 = df['TotalSpending'].quantile(0.75)
df['HVC'] = (df['TotalSpending'] > threshold_75).astype(int)

print("="*80)
print("ANALISIS DISTRIBUSI TARGET VARIABLE: HIGH VALUE CUSTOMER (HVC)")
print("="*80)

# ============================================================================
# 1. JUMLAH CUSTOMER HVC vs NON-HVC
# ============================================================================
print("\n📊 1. JUMLAH CUSTOMER PER KATEGORI")
print("-"*80)

hvc_counts = df['HVC'].value_counts().sort_index()
total_customers = len(df)

print(f"\nTotal Customer: {total_customers:,}\n")
print(f"{'Kategori':<25} {'Jumlah': >10} {'Persentase':>15}")
print("-"*55)

for label, count in hvc_counts. items():
    kategori = 'Non-HVC (0)' if label == 0 else 'HVC (1)'
    persentase = (count / total_customers) * 100
    print(f"{kategori:<25} {count:>10,} {persentase: >14.2f}%")

print("-"*55)

# Hitung rasio
non_hvc_count = hvc_counts[0]
hvc_count = hvc_counts[1]
rasio = non_hvc_count / hvc_count

print(f"\n📌 Rasio Non-HVC :  HVC = {rasio:.2f} :  1")
print(f"   Artinya:  Setiap 1 HVC, ada {rasio:.2f} Non-HVC")

# ============================================================================
# 2. PERSENTASE KELAS (PIE CHART)
# ============================================================================
print("\n\n📊 2. PERSENTASE KELAS")
print("-"*80)

# Statistik per kelas
for label in [0, 1]:
    kategori = 'Non-HVC' if label == 0 else 'HVC'
    count = hvc_counts[label]
    pct = (count / total_customers) * 100
    
    # Statistik spending per kelas
    spending_data = df[df['HVC'] == label]['TotalSpending']
    avg_spending = spending_data.mean()
    min_spending = spending_data.min()
    max_spending = spending_data.max()
    total_revenue = spending_data.sum()
    
    print(f"\n{kategori} ({label}):")
    print(f"  • Jumlah: {count:,} customers ({pct:.2f}%)")
    print(f"  • Avg Spending: £{avg_spending: ,.2f}")
    print(f"  • Min Spending:  £{min_spending:,.2f}")
    print(f"  • Max Spending: £{max_spending:,.2f}")
    print(f"  • Total Revenue: £{total_revenue:,.2f} ({total_revenue/df['TotalSpending'].sum()*100:.1f}% dari total)")

# ============================================================================
# 3. APAKAH DATASET SEIMBANG?
# ============================================================================
print("\n\n❓ 3. APAKAH DATASET SEIMBANG?")
print("-"*80)

# Definisi balance berdasarkan aturan umum
# - Balanced: 40-60% untuk kelas minority
# - Slight imbalance: 20-40% atau 60-80%
# - Moderate imbalance: 10-20% atau 80-90%
# - Severe imbalance: <10% atau >90%

minority_pct = (min(hvc_counts) / total_customers) * 100
majority_pct = (max(hvc_counts) / total_customers) * 100

print(f"Kelas Minority:  {minority_pct:.2f}%")
print(f"Kelas Majority: {majority_pct:.2f}%")
print(f"Rasio Imbalance: {rasio:.2f}: 1\n")

if 40 <= minority_pct <= 60:
    status = "✅ BALANCED (Seimbang)"
    rekomendasi = "Dataset seimbang.  Tidak perlu teknik balancing khusus."
    color_status = "green"
elif 25 <= minority_pct < 40 or 60 < minority_pct <= 75:
    status = "⚠️ SLIGHTLY IMBALANCED (Sedikit Tidak Seimbang)"
    rekomendasi = "Dataset sedikit tidak seimbang.  Bisa gunakan stratified sampling saat train-test split."
    color_status = "yellow"
elif 10 <= minority_pct < 25 or 75 < minority_pct <= 90:
    status = "⚠️ MODERATELY IMBALANCED (Cukup Tidak Seimbang)"
    rekomendasi = "Dataset cukup tidak seimbang. Pertimbangkan:\n" + \
                 "   - Stratified sampling\n" + \
                 "   - Class weight balancing\n" + \
                 "   - SMOTE (Synthetic Minority Over-sampling)"
    color_status = "orange"
else:
    status = "❌ SEVERELY IMBALANCED (Sangat Tidak Seimbang)"
    rekomendasi = "Dataset sangat tidak seimbang.  WAJIB gunakan:\n" + \
                 "   - SMOTE atau ADASYN\n" + \
                 "   - Class weight balancing\n" + \
                 "   - Evaluasi menggunakan F1-Score, Precision-Recall, bukan hanya Accuracy"
    color_status = "red"

print(f"Status: {status}\n")
print(f"💡 Rekomendasi:")
print(f"   {rekomendasi}")

# Catatan tentang threshold persentil 75
print(f"\n📝 Catatan:")
print(f"   Dataset ini menggunakan threshold persentil ke-75, sehingga:")
print(f"   - 75% customer = Non-HVC")
print(f"   - 25% customer = HVC")
print(f"   Ini adalah desain yang disengaja (by design), bukan masalah.")

# ============================================================================
# 4. VISUALISASI
# ============================================================================
print("\n\n📊 4. VISUALISASI DISTRIBUSI")
print("-"*80)

# Set style
sns.set_style('whitegrid')

# Warna konsisten
colors = ['#3498db', '#e74c3c']  # Biru untuk Non-HVC, Merah untuk HVC
labels_text = ['Non-HVC (0)', 'HVC (1)']

# ------------------------
# A. Bar Chart - Jumlah Customer
# ------------------------
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(111)
bars = ax1.bar(labels_text, hvc_counts.values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Kategori Customer', fontsize=12, fontweight='bold')
ax1.set_ylabel('Jumlah Customer', fontsize=12, fontweight='bold')
ax1.set_title('Distribusi Jumlah Customer', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, count in zip(bars, hvc_counts.values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
            f'{count:,}\n({count/total_customers*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('hvc_bar_chart.png', dpi=300, bbox_inches='tight')
print("✅ Visualisasi 1 disimpan: hvc_bar_chart.png")
plt.close()

# ------------------------
# B. Pie Chart - Persentase
# ------------------------
fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111)
explode = (0.05, 0.05)  # Slight separation
wedges, texts, autotexts = ax2.pie(hvc_counts.values, 
                                     labels=labels_text,
                                     colors=colors,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     explode=explode,
                                     shadow=True,
                                     textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('Persentase Kelas HVC vs Non-HVC', fontsize=14, fontweight='bold')

# Make percentage text larger
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('hvc_pie_chart.png', dpi=300, bbox_inches='tight')
print("✅ Visualisasi 2 disimpan: hvc_pie_chart.png")
plt.close()

# ------------------------
# C. Horizontal Bar Chart dengan Persentase
# ------------------------
fig3 = plt.figure(figsize=(10, 6))
ax3 = fig3.add_subplot(111)
y_pos = np.arange(len(labels_text))
bars = ax3.barh(y_pos, hvc_counts.values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels_text)
ax3.set_xlabel('Jumlah Customer', fontsize=12, fontweight='bold')
ax3.set_title('Distribusi Customer (Horizontal)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, count) in enumerate(zip(bars, hvc_counts.values)):
    width = bar.get_width()
    ax3.text(width * 1.02, bar.get_y() + bar.get_height()/2.,
            f' {count:,} ({count/total_customers*100:.1f}%)',
            ha='left', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('hvc_horizontal_bar.png', dpi=300, bbox_inches='tight')
print("✅ Visualisasi 3 disimpan: hvc_horizontal_bar.png")
plt.close()

# ------------------------
# D. Revenue Contribution
# ------------------------
fig4 = plt.figure(figsize=(10, 6))
ax4 = fig4.add_subplot(111)
revenue_by_class = df.groupby('HVC')['TotalSpending'].sum()
revenue_pct = (revenue_by_class / df['TotalSpending'].sum()) * 100

bars = ax4.bar(labels_text, revenue_by_class.values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
ax4.set_xlabel('Kategori Customer', fontsize=12, fontweight='bold')
ax4.set_ylabel('Total Revenue (£)', fontsize=12, fontweight='bold')
ax4.set_title('Kontribusi Revenue per Kategori', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, revenue, pct in zip(bars, revenue_by_class.values, revenue_pct.values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
            f'£{revenue:,.0f}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('hvc_revenue_contribution.png', dpi=300, bbox_inches='tight')
print("✅ Visualisasi 4 disimpan: hvc_revenue_contribution.png")
plt.close()

# ------------------------
# E. Count Plot dengan Seaborn
# ------------------------
fig5 = plt.figure(figsize=(10, 6))
ax5 = fig5.add_subplot(111)
sns.countplot(data=df, x='HVC', hue='HVC', palette=colors, ax=ax5, edgecolor='black', linewidth=1.5, legend=False)
ax5.set_xlabel('HVC Label', fontsize=12, fontweight='bold')
ax5.set_ylabel('Count', fontsize=12, fontweight='bold')
ax5.set_title('Count Plot - HVC Distribution', fontsize=14, fontweight='bold')
ax5.set_xticklabels(['Non-HVC (0)', 'HVC (1)'])
ax5.grid(True, alpha=0.3, axis='y')

# Add count labels
for i, (label, count) in enumerate(hvc_counts.items()):
    ax5.text(i, count * 1.02, f'{count:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('hvc_count_plot.png', dpi=300, bbox_inches='tight')
print("✅ Visualisasi 5 disimpan: hvc_count_plot.png")
plt.close()

# ------------------------
# F. Balance Status Indicator
# ------------------------
fig6 = plt.figure(figsize=(10, 8))
ax6 = fig6.add_subplot(111)
ax6.axis('off')

# Title
ax6.text(0.5, 0.95, 'ANALISIS KESEIMBANGAN DATASET', 
         ha='center', va='top', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Status box
status_color_map = {
    'green': '#2ecc71',
    'yellow': '#f39c12',
    'orange': '#e67e22',
    'red': '#e74c3c'
}

ax6.text(0.5, 0.75, status, 
         ha='center', va='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=status_color_map.get(color_status, 'gray'), 
                   alpha=0.7, edgecolor='black', linewidth=2))

# Metrics
metrics_text = f"""
Kelas Minority: {minority_pct:.2f}%
Kelas Majority: {majority_pct:.2f}%
Rasio Imbalance: {rasio:.2f}:1

Total Customer: {total_customers:,}
• Non-HVC: {non_hvc_count:,} ({non_hvc_count/total_customers*100:.1f}%)
• HVC: {hvc_count:,} ({hvc_count/total_customers*100:.1f}%)
"""

ax6.text(0.5, 0.45, metrics_text, 
         ha='center', va='center', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

# Recommendation
ax6.text(0.5, 0.08, f"💡 {rekomendasi.split(chr(10))[0]}", 
         ha='center', va='bottom', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('hvc_balance_status.png', dpi=300, bbox_inches='tight')
print("✅ Visualisasi 6 disimpan: hvc_balance_status.png")
plt.close()

print("\n✅ Semua visualisasi berhasil disimpan sebagai file terpisah!")

# ============================================================================
# 5. SUMMARY TABLE
# ============================================================================
print("\n\n📋 5. SUMMARY TABLE")
print("-"*80)

summary_df = pd.DataFrame({
    'Kategori': ['Non-HVC (0)', 'HVC (1)', 'TOTAL'],
    'Jumlah Customer': [
        non_hvc_count,
        hvc_count,
        total_customers
    ],
    'Persentase (%)': [
        f"{non_hvc_count/total_customers*100:.2f}%",
        f"{hvc_count/total_customers*100:.2f}%",
        "100.00%"
    ],
    'Avg Spending (£)': [
        f"{df[df['HVC']==0]['TotalSpending']. mean():.2f}",
        f"{df[df['HVC']==1]['TotalSpending'].mean():.2f}",
        f"{df['TotalSpending'].mean():.2f}"
    ],
    'Total Revenue (£)': [
        f"{df[df['HVC']==0]['TotalSpending'].sum():,.2f}",
        f"{df[df['HVC']==1]['TotalSpending'].sum():,.2f}",
        f"{df['TotalSpending']. sum():,.2f}"
    ],
    'Revenue Contribution (%)': [
        f"{df[df['HVC']==0]['TotalSpending'].sum()/df['TotalSpending']. sum()*100:.1f}%",
        f"{df[df['HVC']==1]['TotalSpending'].sum()/df['TotalSpending'].sum()*100:.1f}%",
        "100.0%"
    ]
})

print(summary_df.to_string(index=False))

# ============================================================================
# 6. KESIMPULAN
# ============================================================================
print("\n\n" + "="*80)
print("🎯 KESIMPULAN")
print("="*80)

print(f"""
1. Dataset memiliki {total_customers:,} customer dengan distribusi:
   • Non-HVC: {non_hvc_count:,} ({non_hvc_count/total_customers*100:.1f}%)
   • HVC: {hvc_count:,} ({hvc_count/total_customers*100:.1f}%)

2. Status Keseimbangan:  {status}

3. Meskipun jumlah HVC hanya {hvc_count/total_customers*100:.1f}% dari total customer,
   mereka berkontribusi {df[df['HVC']==1]['TotalSpending'].sum()/df['TotalSpending']. sum()*100:.1f}% dari total revenue. 
   
4. Ini memvalidasi Pareto Principle (80/20 rule): 
   Top {hvc_count/total_customers*100:.0f}% customer berkontribusi signifikan terhadap revenue.

5. Untuk modelling: 
   • Gunakan stratified sampling pada train-test split
   • Evaluasi dengan F1-Score dan Precision-Recall, bukan hanya Accuracy
   • Pertimbangkan class weight balancing pada model
""")

print("="*80)
print("✅ ANALISIS DISTRIBUSI HVC SELESAI!")
print("="*80)