import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import get_HEP_data

def test_data_structure():
    """
    Verifies that data loading functions return DataFrames with expected columns.
    """
    particles = get_HEP_data.get_particles()
    tracker_hits = get_HEP_data.get_tracker_hits()
    calo_hits = get_HEP_data.get_calo_hits()
    
    # Check Particles
    assert isinstance(particles, pd.DataFrame)
    assert 'particle_id' in particles.columns
    assert 'vx' in particles.columns
    assert 'vy' in particles.columns
    
    # Check Tracker Hits
    assert isinstance(tracker_hits, pd.DataFrame)
    assert 'particle_id' in tracker_hits.columns
    assert 'x' in tracker_hits.columns
    
    # Check Calo Hits
    assert isinstance(calo_hits, pd.DataFrame)
    assert 'contrib_particle_ids' in calo_hits.columns
    assert 'total_energy' in calo_hits.columns

def test_hit_ids():
    """
    Verifies that HIT_ID (index) is handled correctly.
    """
    tracker_hits = get_HEP_data.get_tracker_hits()
    calo_hits = get_HEP_data.get_calo_hits()
    
    # Check Tracker Hits Index
    # Should be named 'HIT_ID'
    assert tracker_hits.index.name == 'HIT_ID'
    # Index should be numeric
    assert pd.api.types.is_numeric_dtype(tracker_hits.index)
    
    # Check Calo Hits Index
    assert calo_hits.index.name == 'HIT_ID'
    assert pd.api.types.is_numeric_dtype(calo_hits.index)
    
    # Calo hits are known to have duplicates (one hit -> multiple particles)
    # So index.is_unique should be False (or at least allowed to be False)
    # We just verify that we have an index.
    assert len(calo_hits.index) > 0

def test_data_types():
    """
    Verifies that numeric columns are actually numeric.
    """
    particles = get_HEP_data.get_particles()
    tracker_hits = get_HEP_data.get_tracker_hits()
    
    # Particles
    assert pd.api.types.is_numeric_dtype(particles['particle_id'])
    assert pd.api.types.is_numeric_dtype(particles['vx'])
    
    # Tracker Hits
    assert pd.api.types.is_numeric_dtype(tracker_hits['x'])
    # particle_id might be float or int, but should be numeric
    assert pd.api.types.is_numeric_dtype(tracker_hits['particle_id'])
