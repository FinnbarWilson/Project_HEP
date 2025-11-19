import get_HEP_data
import association_matrix

def test_full_integration():
    """
    Tests the full pipeline with real HEP data.
    """
    print("Loading real HEP data...")
    particles = get_HEP_data.get_particles()
    tracker_hits = get_HEP_data.get_tracker_hits()
    calo_hits = get_HEP_data.get_calo_hits()
    
    print("\nRunning sparse_association_matrix with parent mapping (radius=2.0)...")
    
    tracker_matrix, calo_matrix, unique_pids, pid_map = association_matrix.sparse_association_matrix(
        particles, tracker_hits, calo_hits, radius_threshold=2.0
    )
    
    # Check if matrices are created and have content
    # Note: Matrix rows = max(index) + 1, which might differ from len(df) if index is duplicated (exploded)
    # or if there are gaps.
    print(f"Tracker Hits: len={len(tracker_hits)}, max_idx={tracker_hits.index.max()}")
    print(f"Tracker Matrix: shape={tracker_matrix.shape}")
    
    print(f"Calo Hits: len={len(calo_hits)}, max_idx={calo_hits.index.max()}")
    print(f"Calo Matrix: shape={calo_matrix.shape}")

    assert tracker_matrix.shape[0] == tracker_hits.index.max() + 1, \
        f"Tracker matrix row count mismatch: {tracker_matrix.shape[0]} vs {tracker_hits.index.max() + 1}"
        
    assert calo_matrix.shape[0] == calo_hits.index.max() + 1, \
        f"Calo matrix row count mismatch: {calo_matrix.shape[0]} vs {calo_hits.index.max() + 1}"
    assert len(unique_pids) > 0, "No unique particles found"
    
    assert tracker_matrix.nnz > 0, "Tracker matrix is empty"
    assert calo_matrix.nnz > 0, "Calo matrix is empty"
