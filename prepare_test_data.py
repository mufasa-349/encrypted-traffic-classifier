"""Prepare test data by removing labels."""
import pandas as pd
import sys

csv_path = 'data/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
output_no_labels = 'test_data_no_labels.csv'
output_labels = 'test_data_labels_backup.csv'

print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path, low_memory=False)
print(f"Original: {len(df)} rows, {len(df.columns)} cols")

# Save labels
labels = df['Label'].copy()
labels.to_csv(output_labels, index=False)
print(f"Labels saved to {output_labels}")

# Remove labels
df_no_label = df.drop(columns=['Label'])
df_no_label.to_csv(output_no_labels, index=False)
print(f"Data without labels saved to {output_no_labels}")
print(f"Without labels: {len(df_no_label)} rows, {len(df_no_label.columns)} cols")
print(f"Labels saved: {len(labels)} labels")

