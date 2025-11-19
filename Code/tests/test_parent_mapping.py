import pandas as pd
import association_matrix

def test_parent_mapping_logic():
    """
    Tests the map_daughters_to_parents function with mock data.
    """
    # Mock Particles
    # P1: Parent (r=0)
    # P2: Daughter of P1 (r=20 > 10)
    # P3: Granddaughter of P1 (daughter of P2) (r=30 > 10)
    # P4: Parent (r=5)
    # P5: Daughter of P4 (r=15 > 10)
    # P6: Noise/Orphan (r=50, parent=0) -> Should map to itself if 0 is not in list, or 0.
    
    particles_data = {
        'particle_id': [1, 2, 3, 4, 5, 6],
        'parent_id':   [0, 1, 2, 0, 4, 0],
        'vx':          [0, 20, 30, 3, 15, 50],
        'vy':          [0, 0,  0,  4, 0,  0]
    }
    particles = pd.DataFrame(particles_data)
    
    # Mock Hits
    # H1 -> P1 (Parent) -> Should stay P1
    # H2 -> P2 (Daughter) -> Should become P1
    # H3 -> P3 (Granddaughter) -> Should become P1
    # H4 -> P5 (Daughter) -> Should become P4
    # H5 -> P6 (Orphan) -> Should stay P6 (as it has no parent inside radius)
    
    hits_data = {
        'hit_id': [1, 2, 3, 4, 5],
        'particle_id': [1, 2, 3, 5, 6]
    }
    hits = pd.DataFrame(hits_data)
    
    mapped_hits = association_matrix.map_daughters_to_parents(
        particles, hits, 'particle_id', radius_threshold=10.0
    )
    
    expected_pids = [1, 1, 1, 4, 0]
    actual_pids = mapped_hits['particle_id'].tolist()
    
    assert expected_pids == actual_pids, f"Expected {expected_pids}, but got {actual_pids}"
