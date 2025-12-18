import pandas as pd
import numpy as np
import glob
import os

DATA_DIR = '/Users/finn/Documents/Code/Large_Datasets/TrackML/train_100_events'
hit_files = sorted(glob.glob(os.path.join(DATA_DIR, '*-hits.csv')))
event_prefix = hit_files[0].replace('-hits.csv', '')
particles = pd.read_csv(f"{event_prefix}-particles.csv")

# Extract unique vertices
unique_v = particles[['vx', 'vy', 'vz']].drop_duplicates()
unique_v['R'] = np.sqrt(unique_v['vx']**2 + unique_v['vy']**2)

print(f"Total Unique Vertices: {len(unique_v)}")
print(f"Vertices with R < 1mm: {len(unique_v[unique_v['R'] < 1.0])}")
print(f"Vertices with R < 5mm: {len(unique_v[unique_v['R'] < 5.0])}")
print(f"Vertices with R > 5mm: {len(unique_v[unique_v['R'] > 5.0])}")

print("\nSample High-R Vertices (Secondary decays?):")
print(unique_v[unique_v['R'] > 50.0].head())

print("\nSample Low-R Vertices (Primaries?):")
print(unique_v[unique_v['R'] < 1.0].head())
