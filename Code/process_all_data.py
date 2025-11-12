import json
import numpy as np
import pandas as pd
import get_HEP_data as gHd

all_events_data = {}

for n in range(10):
    # 1. Retreaving data
    particles = gHd.get_particles()
    tracker_hits = gHd.get_tracker_hits()
    calo_hits = gHd.get_calo_hits()
    
    particles = particles[particles['event_id'] == n].copy()
    tracker_hits = tracker_hits[tracker_hits['event_id'] == n].copy()
    calo_hits = calo_hits[calo_hits['event_id'] == n].copy()

    # 2. Apply static cuts
    particles = particles[abs(particles['vx']) < 1]
    particles = particles[abs(particles['vy']) < 1]
    
    # 3. Calculate pT and PDG ID
    particles['pT'] = np.sqrt(particles['px']**2 + particles['py']**2)
    
    # Get particle IDs to match
    particles_id = particles["particle_id"].unique()
    
    # Create lookup maps for particle_id -> pT and particle_id -> pdg_id
    pt_map = particles.set_index('particle_id')['pT'].to_dict()
    pdg_map = particles.set_index('particle_id')['pdg_id'].to_dict()

    # 4.1 Process ALL Tracker Hits (for the point cloud)
    # Filter to only include hits from particles we're interested in (after static cuts)
    all_event_tracker_hits = []
    if not tracker_hits.empty:
        filtered_tracker_hits = tracker_hits[tracker_hits["particle_id"].isin(particles_id)].copy()
        
        # Add pT and pdg_id to each hit for filtering/coloring in JS
        filtered_tracker_hits['pT'] = filtered_tracker_hits['particle_id'].map(pt_map)
        filtered_tracker_hits['pdg_id'] = filtered_tracker_hits['particle_id'].map(pdg_map)
        filtered_tracker_hits = filtered_tracker_hits.dropna(subset=['pT', 'pdg_id']) # Drop hits without a pT or PDG ID
        
        all_event_tracker_hits = filtered_tracker_hits[['x', 'y', 'z', 'pT', 'pdg_id', 'particle_id']].to_dict(orient='records')


    # 4.2 Process Tracks (Lines) - same as before, but ensure pT and pdg_id are added
    tracker_hits_for_tracks = tracker_hits[tracker_hits["particle_id"].isin(particles_id)].copy() # Use a copy to avoid SettingWithCopyWarning
    tracker_hits_for_tracks['r'] = np.sqrt(tracker_hits_for_tracks['x']**2 + tracker_hits_for_tracks['y']**2 + tracker_hits_for_tracks['z']**2)
    tracker_hits_for_tracks = tracker_hits_for_tracks.sort_values(by=['particle_id', 'r'])

    event_tracks = []
    for particle_id, group_of_hits in tracker_hits_for_tracks.groupby('particle_id'):
        points = group_of_hits[['x', 'y', 'z']].to_dict(orient='records')
        
        if len(points) > 1:
            event_tracks.append({
                'particle_id': int(particle_id),
                'pT': float(pt_map.get(particle_id, 0)),
                'pdg_id': int(pdg_map.get(particle_id, 0)),
                'points': points
            })

    # 5. Process Calo Hits - same as before, but ensure pdg_id is added
    event_calo_hits = []
    if 'contrib_particle_ids' in calo_hits.columns and not calo_hits.empty:
        calo_hits = calo_hits.explode(['contrib_particle_ids', 'contrib_energies'])
        calo_hits = calo_hits.dropna(subset=['contrib_particle_ids'])
        
        calo_hits['contrib_particle_ids'] = pd.to_numeric(calo_hits['contrib_particle_ids'])
        calo_hits = calo_hits[calo_hits['contrib_particle_ids'].isin(particles_id)]
        
        calo_hits['pT'] = calo_hits['contrib_particle_ids'].map(pt_map)
        calo_hits['pdg_id'] = calo_hits['contrib_particle_ids'].map(pdg_map)
        calo_hits = calo_hits.dropna(subset=['pT', 'pdg_id'])
        
        calo_hits['contrib_energies'] = pd.to_numeric(calo_hits['contrib_energies'])

        temp_calo_hits = calo_hits[['x', 'y', 'z', 'contrib_energies', 'pT', 'pdg_id', 'contrib_particle_ids']].to_dict(orient='records')
        
        event_calo_hits = [
            {
                'x': h['x'], 'y': h['y'], 'z': h['z'],
                'energy': h['contrib_energies'],
                'pT': h['pT'],
                'pdg_id': int(h['pdg_id']),
                'particle_id': int(h['contrib_particle_ids'])
            } for h in temp_calo_hits
        ]

    # 6. Store both in the main dictionary
    all_events_data[n] = {
        'tracks': event_tracks,
        'calo_hits': event_calo_hits,
        'all_tracker_hits': all_event_tracker_hits # NEW: Add all tracker hits
    }
    
    print(f"Event {n}: Processed {len(event_tracks)} tracks, {len(event_calo_hits)} calo hits, and {len(all_event_tracker_hits)} raw tracker hits.")

# 7. Save to the JSON file
output_filename = 'data/event_data_full.json'
with open(output_filename, 'w') as f:
    json.dump(all_events_data, f, indent=4)

print(f"\nAll event data saved to {output_filename}")