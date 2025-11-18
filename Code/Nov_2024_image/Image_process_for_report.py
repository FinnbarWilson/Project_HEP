import h5py
import json
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = '/Users/finn/Documents/Large_Datasets/epoch=028-val_loss=1.29786__test_test.h5'
OUTPUT_FILE = 'Nov_2024_image/Nov_2024_event_data_full.json'

# REDUCTION FACTOR
# 1 = All tracks
# 10 = Only 1 in every 10 tracks (Cleaner view, faster load)
TRACK_DECIMATION = 20

# --- VIRTUAL DETECTOR GEOMETRY (mm) ---
MAX_R = 1200.0  # Barrel Radius
MAX_Z = 3000.0  # Endcap Z
B_FIELD = 3.8   # Tesla

def get_bounded_path(pt, eta, phi, vz, charge, step_size_mm=20):
    """ Simulates particle flight until it hits detector walls. """
    if pt < 0.1: return [] 
    
    # equation from http://lppp.lancs.ac.uk/motioninb/en-GB/experiment.html?LPPPSession=1761782400010
    R = (pt / (0.3 * B_FIELD)) * 1000.0 # 1000 to convert to mm
    curvature = 1.0 / R
    slope_z = np.sinh(eta)
    
    points = []
    s = 0.0
    max_steps = 500 
    
    # Track direction
    curvature_sign = -1 if charge < 0 else 1

    for i in range(max_steps):
        # Angle deflection based on arc length s
        alpha = s * curvature * curvature_sign
        
        # Calculate position
        # Geometric shift for circle starting at 0,0 tangent to phi
        dx = R * (np.sin(phi + alpha) - np.sin(phi))
        dy = R * (np.cos(phi) - np.cos(phi + alpha))

        x = dx
        y = dy
        z = vz + (s * slope_z)
        
        current_r = np.sqrt(x**2 + y**2)
        
        # Check boundaries
        if current_r >= MAX_R:
            points.append({"x": x, "y": y, "z": z})
            break
        if abs(z) >= MAX_Z:
            points.append({"x": x, "y": y, "z": z})
            break
        
        points.append({"x": float(x), "y": float(y), "z": float(z)})
        s += step_size_mm

    return points

def convert_h5_sparse():
    all_events_data = {}
    
    try:
        f = h5py.File(INPUT_FILE, 'r')
    except Exception as e:
        print(f"Error: {e}")
        return

    event_keys = sorted([k for k in f.keys() if k.startswith('event_')])[:40]
    print(f"Processing {len(event_keys)} events (Decimation: 1 in {TRACK_DECIMATION})...")

    for idx, event_name in enumerate(event_keys):
        try:
            parts = f[event_name]['parts']
            pts = parts['pts'][:]
            etas = parts['etas'][:]
            phis = parts['phis'][:]
            pids = parts['pids'][:]
            vzs = parts['vzs'][:]
        except KeyError:
            continue

        event_tracks = []
        total_tracks_in_file = len(pts)
        
        # --- THE CHANGE IS HERE ---
        # We step through the arrays by TRACK_DECIMATION (e.g., 0, 10, 20, 30...)
        for i in range(0, total_tracks_in_file, TRACK_DECIMATION):
            
            pt = float(pts[i])
            huge_id = int(pids[i])
            
            # Still apply physics cut
            if pt < 0.5: continue
            
            # Infer properties
            pdg_id = 211 
            charge = 1.0 if huge_id % 2 == 0 else -1.0
            
            # Generate Path
            path_points = get_bounded_path(pt, float(etas[i]), float(phis[i]), float(vzs[i]), charge)
            
            if len(path_points) > 1:
                event_tracks.append({
                    "particle_id": str(huge_id),
                    "pT": pt,
                    "pdg_id": pdg_id,
                    "points": path_points
                })

        all_events_data[idx] = {
            "tracks": event_tracks,
            "calo_hits": [], 
            "all_tracker_hits": [] 
        }
        print(f"  {event_name}: Reduced {total_tracks_in_file} -> {len(event_tracks)} tracks")

    with open(OUTPUT_FILE, 'w') as outfile:
        json.dump(all_events_data, outfile, indent=4)
        
    print(f"Saved sparse data to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_h5_sparse()