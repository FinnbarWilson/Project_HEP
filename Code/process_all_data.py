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
    
    # 3. Calculate pT
    particles['pT'] = np.sqrt(particles['px']**2 + particles['py']**2)
    
    # Get particle IDs to match
    particles_id = particles["particle_id"].unique()
    
    # Create a lookup map for particle_id -> pT
    pt_map = particles.set_index('particle_id')['pT'].to_dict()

    # 4. Process Tracker Hits
    tracker_hits = tracker_hits[tracker_hits["particle_id"].isin(particles_id)]
    tracker_hits['r'] = np.sqrt(tracker_hits['x']**2 + tracker_hits['y']**2 + tracker_hits['z']**2)
    tracker_hits = tracker_hits.sort_values(by=['particle_id', 'r'])

    event_tracks = []
    for particle_id, group_of_hits in tracker_hits.groupby('particle_id'):
        points = group_of_hits[['x', 'y', 'z']].to_dict(orient='records')
        
        if len(points) > 1:
            event_tracks.append({
                'particle_id': int(particle_id),
                'pT': float(pt_map.get(particle_id, 0)), # Add pT to each track
                'points': points
            })

    # 5. Process Calo Hits
    event_calo_hits = []
    # Check if calo data exists and has the required columns
    if 'contrib_particle_ids' in calo_hits.columns and not calo_hits.empty:
        calo_hits = calo_hits.explode(['contrib_particle_ids', 'contrib_energies'])
        calo_hits = calo_hits.dropna(subset=['contrib_particle_ids'])
        
        # Filter calo hits that match our selected particles
        calo_hits['contrib_particle_ids'] = pd.to_numeric(calo_hits['contrib_particle_ids'])
        calo_hits = calo_hits[calo_hits['contrib_particle_ids'].isin(particles_id)]
        
        # Add pT to calo hits using the map
        calo_hits['pT'] = calo_hits['contrib_particle_ids'].map(pt_map)
        calo_hits = calo_hits.dropna(subset=['pT']) # Drop hits without a pT
        
        calo_hits['contrib_energies'] = pd.to_numeric(calo_hits['contrib_energies'])

        # Format for JSON
        temp_calo_hits = calo_hits[['x', 'y', 'z', 'contrib_energies', 'pT', 'contrib_particle_ids']].to_dict(orient='records')
        
        event_calo_hits = [
            {
                'x': h['x'], 'y': h['y'], 'z': h['z'],
                'energy': h['contrib_energies'],
                'pT': h['pT'],
                'particle_id': int(h['contrib_particle_ids'])
            } for h in temp_calo_hits
        ]

    # 6. Store both in the main dictionary
    all_events_data[n] = {
        'tracks': event_tracks,
        'calo_hits': event_calo_hits
    }
    
    print(f"Event {n}: Processed {len(event_tracks)} tracks and {len(event_calo_hits)} calo hits.")

# 7. Save to a NEW JSON file
output_filename = 'data/event_data_full.json'
with open(output_filename, 'w') as f:
    json.dump(all_events_data, f, indent=4)

print(f"\nAll event data saved to {output_filename}")