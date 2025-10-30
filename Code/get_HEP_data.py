import pandas as pd

def get_calo_hits():
    calo_hits = pd.read_parquet('data/parquet/reco/calo_hits/hard_scatter.ttbar.v1.reco.calo_hits.events0-9.parquet')
    calo_hits = calo_hits.explode([col for col in calo_hits.columns if col != 'event_id'])
    calo_hits = calo_hits.apply(pd.to_numeric)
    return calo_hits

def get_tracker_hits():
    tracker_hits = pd.read_parquet('data/parquet/reco/tracker_hits/hard_scatter.ttbar.v1.reco.tracker_hits.events0-9.parquet')
    tracker_hits = tracker_hits.explode([col for col in tracker_hits.columns if col != 'event_id'])
    tracker_hits = tracker_hits.apply(pd.to_numeric)
    return tracker_hits

def get_tracks():
    tracks = pd.read_parquet('data/parquet/reco/tracks/hard_scatter.ttbar.v1.reco.tracks.events0-9.parquet')
    tracks = tracks.explode([col for col in tracks.columns if col != 'event_id'])
    tracks = tracks.apply(pd.to_numeric)
    return tracks

def get_particles():
    particles = pd.read_parquet('data/parquet/truth/particles/hard_scatter.ttbar.v1.truth.particles.events0-9.parquet')
    particles = particles.explode([col for col in particles.columns if col != 'event_id'])
    particles = particles.apply(pd.to_numeric)
    return particles