import pandas as pd
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def get_calo_hits():
    path = os.path.join(DATA_DIR, 'parquet/reco/calo_hits/hard_scatter.ttbar.v1.reco.calo_hits.events0-9.parquet')
    calo_hits = pd.read_parquet(path)
    calo_hits = calo_hits.explode([col for col in calo_hits.columns if col != 'event_id'])
    # Create unique ID for each hit
    calo_hits = calo_hits.reset_index(drop=True) 
    calo_hits = calo_hits.reset_index().rename(columns={'index': 'HIT_ID'}) # Now has a 'HIT_ID' column (0, 1, 2...)
    calo_hits = calo_hits.explode(['contrib_particle_ids', 'contrib_energies', 'contrib_times'])
    # Set the (now duplicated) HIT_ID as the official DataFrame index
    calo_hits = calo_hits.set_index('HIT_ID')
    all_numeric_cols = ['cell_id', 'total_energy', 'x', 'y', 'z','contrib_particle_ids', 'contrib_energies', 'contrib_times']
    for col in all_numeric_cols:
        calo_hits[col] = pd.to_numeric(calo_hits[col], errors='coerce')
    return calo_hits

def get_tracker_hits():
    path = os.path.join(DATA_DIR, 'parquet/reco/tracker_hits/hard_scatter.ttbar.v1.reco.tracker_hits.events0-9.parquet')
    tracker_hits = pd.read_parquet(path)
    tracker_hits = tracker_hits.explode([col for col in tracker_hits.columns if col != 'event_id'])
    tracker_hits = tracker_hits.apply(pd.to_numeric, errors='coerce')
    tracker_hits = tracker_hits.reset_index(drop=True)
    return tracker_hits

def get_tracks():
    path = os.path.join(DATA_DIR, 'parquet/reco/tracks/hard_scatter.ttbar.v1.reco.tracks.events0-9.parquet')
    tracks = pd.read_parquet(path)
    tracks = tracks.explode([col for col in tracks.columns if col != 'event_id'])
    tracks = tracks.explode('hit_ids')
    tracks = tracks.apply(pd.to_numeric, errors='coerce')
    tracks = tracks.reset_index(drop=True)
    return tracks

def get_particles():
    path = os.path.join(DATA_DIR, 'parquet/truth/particles/hard_scatter.ttbar.v1.truth.particles.events0-9.parquet')
    particles = pd.read_parquet(path)
    particles = particles.explode([col for col in particles.columns if col != 'event_id'])
    particles = particles.apply(pd.to_numeric, errors='coerce')
    particles = particles.reset_index(drop=True)
    return particles