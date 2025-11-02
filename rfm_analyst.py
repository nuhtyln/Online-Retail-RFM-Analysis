import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

print("Langkah 1: Memuat dan Membersihkan Data...")

try:
    df = pd.read_csv('OnlineRetail.csv', encoding='latin-1')
except FileNotFoundError:
    print("ERROR: File 'OnlineRetail.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    exit()

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

df.dropna(subset=['CustomerID'], inplace=True)
df['CustomerID'] = df['CustomerID'].astype(int)

df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

print(f"Data bersih siap diolah: {df.shape[0]} baris.")

print("\nLangkah 2: Menghitung Nilai Recency, Frequency, dan Monetary...")

snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm_df = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()

print("\nDataFrame RFM (sebelum scoring):")
print(rfm_df.head())

print("\nLangkah 3: Pemberian Skor (Skala 1-5) menggunakan Rank...")

rfm_df['R_Rank'] = rfm_df['Recency'].rank(method='first', ascending=False)
rfm_df['R_Score'] = pd.qcut(
    rfm_df['R_Rank'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop'
).astype(int)

rfm_df['F_Rank'] = rfm_df['Frequency'].rank(method='first', ascending=True)
rfm_df['F_Score'] = pd.qcut(
    rfm_df['F_Rank'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop'
).astype(int)

rfm_df['M_Rank'] = rfm_df['Monetary'].rank(method='first', ascending=True)
rfm_df['M_Score'] = pd.qcut(
    rfm_df['M_Rank'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop'
).astype(int)


rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + \
                      rfm_df['F_Score'].astype(str) + \
                      rfm_df['M_Score'].astype(str)

print("\nDataFrame RFM (setelah scoring):")
print(rfm_df[['CustomerID', 'Recency', 'R_Score', 'Frequency', 'F_Score', 'Monetary', 'M_Score', 'RFM_Score']].head())

print("\nLangkah 4: Segmentasi Pelanggan...")

def rfm_segment(df):
    if df['RFM_Score'] in ['555', '544', '545', '455', '554']:
        return '01 - Champions (Terbaik)'
    elif df['RFM_Score'] in ['543', '444', '434', '344', '355', '454', '453']:
        return '02 - Loyal Customers (Setia)'
    elif df['RFM_Score'] in ['512', '511', '411', '412', '311', '522', '422']:
        return '03 - New/Promising (Baru/Menjanjikan)'
    elif df['RFM_Score'] in ['222', '233', '322', '234', '334', '433']:
        return '04 - Need Attention (Butuh Perhatian)'
    elif df['RFM_Score'] in ['122', '212', '112', '221', '312', '133', '121']:
        return '05 - At Risk (Berisiko Hilang)'
    else:
        return '06 - Lost/Others (Hilang/Lainnya)'

rfm_df['Customer_Segment'] = rfm_df.apply(rfm_segment, axis=1)

segment_summary = rfm_df.groupby('Customer_Segment').agg(
    Count=('CustomerID', 'nunique'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).sort_values(by='Count', ascending=False).reset_index()

print("\n--- RINGKASAN SEGMENTASI RFM ---")
print(segment_summary)

rfm_df.to_csv('rfm_analysis_result_fixed.csv', index=False)
print("\nHasil analisis telah disimpan ke 'rfm_analysis_result_fixed.csv'")

print("\nLangkah 5: Visualisasi Hasil Segmentasi...")

segment_counts = rfm_df['Customer_Segment'].value_counts().reset_index()
segment_counts.columns = ['Customer_Segment', 'Count']

segment_counts = segment_counts.sort_values('Customer_Segment')

plt.figure(figsize=(12, 6))
sns.barplot(
    x='Customer_Segment',
    y='Count',
    data=segment_counts,
    palette='viridis' 
)

for index, row in segment_counts.iterrows():
    plt.text(
        index, 
        row['Count'] + 50, 
        f"{row['Count']}\n({(row['Count']/segment_counts['Count'].sum()*100):.1f}%)",
        color='black',
        ha="center",
        fontsize=10
    )

plt.title('Distribusi Pelanggan Berdasarkan Segmen RFM', fontsize=16)
plt.xlabel('Segmen Pelanggan', fontsize=12)
plt.ylabel('Jumlah Pelanggan', fontsize=12)
plt.xticks(rotation=45, ha='right') 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nVisualisasi telah berhasil ditampilkan.")

print("\nLangkah 6: Visualisasi Scatter Plot RFM...")

segment_order = rfm_df['Customer_Segment'].unique()
segment_order.sort()

plt.figure(figsize=(12, 6))

sns.scatterplot(
    x='Recency', 
    y='Monetary', 
    hue='Customer_Segment', 
    data=rfm_df, 
    palette='viridis', 
    s=50, 
    alpha=0.6, 
    hue_order=segment_order
)

plt.title('Scatter Plot: Recency vs. Monetary Value (Diwarnai berdasarkan Segmen)', fontsize=16)
plt.xlabel('Recency (Hari sejak pembelian terakhir)', fontsize=12)
plt.ylabel('Monetary Value (Total Pengeluaran)', fontsize=12)
plt.legend(title='Segmen Pelanggan', bbox_to_anchor=(1.05, 1), loc=2) 
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1]) 
plt.show()


plt.figure(figsize=(12, 6))

sns.scatterplot(
    x='Frequency', 
    y='Monetary', 
    hue='Customer_Segment', 
    data=rfm_df, 
    palette='viridis',
    s=50,
    alpha=0.6,
    hue_order=segment_order
)

plt.title('Scatter Plot: Frequency vs. Monetary Value (Diwarnai berdasarkan Segmen)', fontsize=16)
plt.xlabel('Frequency (Jumlah Transaksi)', fontsize=12)
plt.ylabel('Monetary Value (Total Pengeluaran)', fontsize=12)
plt.legend(title='Segmen Pelanggan', bbox_to_anchor=(1.05, 1), loc=2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()

print("\nVisualisasi Scatter Plot telah selesai.")