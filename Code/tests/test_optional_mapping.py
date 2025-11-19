import pandas as pd
import association_matrix

def test_optional_mapping():
    """
    Tests the map_daughters flag in sparse_association_matrix.
    """
    # Mock Particles
    # P1: Parent (r=0)
    # P2: Daughter of P1 (r=20 > 2.0)
    particles = pd.DataFrame({
        'particle_id': [1, 2],
        'parent_id':   [0, 1],
        'vx':          [0, 20],
        'vy':          [0, 0]
    })
    
    # Mock Tracker Hits
    tracker_hits = pd.DataFrame({
        'hit_id': [1],
        'particle_id': [2]
    })

    # Mock Calo Hits
    calo_hits = pd.DataFrame({
        'hit_id': [1],
        'contrib_particle_ids': [2]
    })
    
    # Test 1: map_daughters=True
    tracker_matrix, _, _, _ = association_matrix.sparse_association_matrix(
        particles, tracker_hits, calo_hits, map_daughters=True, radius_threshold=2.0
    )
    
    # With mapping, P2 -> P1. P1 is index 0.
    # Matrix should have entry at (0, 0)
    assert tracker_matrix[0, 0] == 1, "Daughter should be mapped to parent when map_daughters=True"
        
    # Test 2: map_daughters=False
    tracker_matrix, _, _, _ = association_matrix.sparse_association_matrix(
        particles, tracker_hits, calo_hits, map_daughters=False, radius_threshold=2.0
    )
    
    # Without mapping, P2 stays P2. P2 is index 1.
    # Matrix should have entry at (0, 1)
    assert tracker_matrix[0, 1] == 1, "Daughter should NOT be mapped when map_daughters=False"
