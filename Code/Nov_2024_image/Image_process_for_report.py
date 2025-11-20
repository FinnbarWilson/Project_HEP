import pandas as pd
import numpy as np
import json
import os
import glob

# --- CONFIGURATION ---
# Path to the directory containing the CSV files
DATA_DIR = '/Users/finn/Documents/Large_Datasets/TrackML/train_100_events'
OUTPUT_FILE = 'Nov_2024_image/Nov_2024_event_data_full.json'

# Filter Configuration
VOLUMES_OF_INTEREST = [7, 8, 9]
MIN_HITS_TOTAL = 3

def process_trackml_data():
    all_events_data = {}
    
    # Find all event prefixes (e.g., event000001000)
    hit_files = sorted(glob.glob(os.path.join(DATA_DIR, '*-hits.csv')))
    
    if not hit_files:
        print(f"No data found in {DATA_DIR}. Please ensure CSV files are present.")
        return

    # Process first 5 events
    for hit_file in hit_files[:5]:
        event_prefix = hit_file.replace('-hits.csv', '')
        event_name = os.path.basename(event_prefix)
        
        print(f"Processing {event_name}...")
        
        try:
            # Load Data
            hits = pd.read_csv(f"{event_prefix}-hits.csv")
            particles = pd.read_csv(f"{event_prefix}-particles.csv")
            truth = pd.read_csv(f"{event_prefix}-truth.csv")
            
            # Merge to get complete picture: Hits + Truth (Particle ID)
            # We only care about hits that are associated with a particle (particle_id != 0)
            hits_truth = pd.merge(hits, truth[['hit_id', 'particle_id', 'weight']], on='hit_id')
            
            # Filter out noise (particle_id == 0)
            hits_truth = hits_truth[hits_truth['particle_id'] != 0]
            
            # Merge with Particle info to get pT, eta, etc.
            # particles df has: particle_id, vx, vy, vz, px, py, pz, q, nhits
            full_data = pd.merge(hits_truth, particles, on='particle_id')
            
            # Calculate pT for filtering/display
            full_data['pT'] = np.sqrt(full_data['px']**2 + full_data['py']**2)
            
            # Group by Particle
            grouped = full_data.groupby('particle_id')
            
            event_tracks = []
            
            # Map particle_id to a smaller integer to save space
            particle_id_map = {}
            next_pid = 0

            for pid, group in grouped:
                # Filter: Min Hits
                if len(group) < MIN_HITS_TOTAL:
                    continue
                
                # Get hits
                # Sort by R (distance from origin) to make drawing easier/sequential
                group['R'] = np.sqrt(group['x']**2 + group['y']**2)
                sorted_group = group.sort_values('R')
                
                track_points = []
                
                for _, row in sorted_group.iterrows():
                    vol_id = int(row['volume_id'])
                    in_roi = vol_id in VOLUMES_OF_INTEREST
                        
                    track_points.append({
                        "x": round(float(row['x']), 2),
                        "y": round(float(row['y']), 2),
                        "z": round(float(row['z']), 2),
                        # "vol_id": vol_id, # Removed to save space
                        "in_roi": 1 if in_roi else 0 # Use 1/0 instead of true/false for slight saving
                    })
                
                # Extract particle properties from the first row (same for all hits of this particle)
                particle_props = group.iloc[0]

                # Remap particle_id
                if pid not in particle_id_map:
                    particle_id_map[pid] = next_pid
                    next_pid += 1
                
                event_tracks.append({
                    "particle_id": particle_id_map[pid],
                    "pT": round(float(particle_props['pT']), 2),
                    "pdg_id": int(particle_props['pdg_id'] if 'pdg_id' in particle_props else 0),
                    "points": track_points
                })
            
            all_events_data[event_name] = {
                "tracks": event_tracks
            }
            print(f"  -> {len(event_tracks)} tracks extracted.")
            
        except Exception as e:
            print(f"  Error processing {event_name}: {e}")
            continue

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as outfile:
        # Use separators to remove whitespace for minification
        json.dump(all_events_data, outfile, separators=(',', ':'))
        
    print(f"Saved data to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_trackml_data()