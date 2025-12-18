import pandas as pd
import numpy as np
import glob
import os

DATA_DIR = '/Users/finn/Documents/Code/Large_Datasets/TrackML/train_100_events'
hit_files = sorted(glob.glob(os.path.join(DATA_DIR, '*-hits.csv')))
event_prefix = hit_files[0].replace('-hits.csv', '')
particles = pd.read_csv(f"{event_prefix}-particles.csv")

print(f"Total Particles: {len(particles)}")
print(f"Unique vz values: {particles['vz'].nunique()}")
print(f"Unique vx values: {particles['vx'].nunique()}")

# Check spacing between sorted unique vz values
sorted_vz = np.sort(particles['vz'].unique())
diffs = np.diff(sorted_vz)
print(f"Min diff: {diffs.min()}, Max diff: {diffs.max()}, Mean diff: {diffs.mean()}")
print(f"Count of diffs < 0.1mm: {np.sum(diffs < 0.1)}")
print(f"Count of diffs > 1.0mm: {np.sum(diffs > 1.0)}")
