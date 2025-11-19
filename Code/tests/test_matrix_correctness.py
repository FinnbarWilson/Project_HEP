import pytest
import numpy as np
import pandas as pd
import get_HEP_data
import association_matrix

def test_matrix_correctness_spot_checks():
    """
    Performs rigorous spot checks to ensure the sparse matrix accurately reflects the input DataFrame.
    
    1. Forward Check: Sample random hits from DataFrame -> Verify they exist in Matrix.
    2. Backward Check: Sample random non-zero entries from Matrix -> Verify they exist in DataFrame.
    """
    print("Loading data...")
    particles = get_HEP_data.get_particles()
    tracker_hits = get_HEP_data.get_tracker_hits()
    calo_hits = get_HEP_data.get_calo_hits()
    
    radius_threshold = 2.0
    
    print("Generating matrices...")
    tracker_matrix, calo_matrix, unique_pids, pid_map = association_matrix.sparse_association_matrix(
        particles, tracker_hits, calo_hits, map_daughters=True, radius_threshold=radius_threshold
    )
    
    # Helper to map a PID to its parent (re-implementing logic for verification)
    
    print("Generating expected mappings for verification...")
    tracker_hits_mapped = association_matrix.map_daughters_to_parents(
        particles, tracker_hits, 'particle_id', radius_threshold
    )
    calo_hits_mapped = association_matrix.map_daughters_to_parents(
        particles, calo_hits, 'contrib_particle_ids', radius_threshold
    )
    
    # --- Forward Check (DataFrame -> Matrix) ---
    print("Running Forward Checks (DataFrame -> Matrix)...")
    n_samples = 100
    
    # Tracker
    print(f"Tracker Index Unique: {tracker_hits_mapped.index.is_unique}")
    if len(tracker_hits) > 0:
        # Sample by integer position (iloc) to avoid ambiguity if index has duplicates
        sample_positions = np.random.choice(len(tracker_hits), size=min(n_samples, len(tracker_hits)), replace=False)
        
        for pos in sample_positions:
            hit_row = tracker_hits_mapped.iloc[pos]
            idx = hit_row.name # The index value
            pid = hit_row['particle_id']
            
            # If PID is not in our map (e.g. noise or filtered out), it shouldn't be in the matrix
            if pid in pid_map:
                col_idx = pid_map[pid]
                assert tracker_matrix[idx, col_idx] == 1, \
                    f"Tracker Hit {idx} with PID {pid} (col {col_idx}) not found in matrix!"

    # Calo
    print(f"Calo Index Unique: {calo_hits_mapped.index.is_unique}")
    if len(calo_hits) > 0:
        sample_positions = np.random.choice(len(calo_hits), size=min(n_samples, len(calo_hits)), replace=False)
        
        for pos in sample_positions:
            hit_row = calo_hits_mapped.iloc[pos]
            idx = hit_row.name # The index value
            pid = hit_row['contrib_particle_ids']
            
            if pid in pid_map:
                col_idx = pid_map[pid]
                val = calo_matrix[idx, col_idx]
                assert val >= 1, \
                    f"Calo Hit {idx} with PID {pid} (col {col_idx}) not found in matrix! Val={val}"

    # --- Backward Check (Matrix -> DataFrame) ---
    print("Running Backward Checks (Matrix -> DataFrame)...")
    
    # Tracker
    if tracker_matrix.nnz > 0:
        # Get all non-zero coordinates
        rows, cols = tracker_matrix.nonzero()
        # Sample random indices
        sample_indices = np.random.choice(len(rows), size=min(n_samples, len(rows)), replace=False)
        
        for i in sample_indices:
            row_idx = rows[i] # Hit ID
            col_idx = cols[i] # Particle Index
            
            # Get PID from col_idx
            pid = unique_pids[col_idx]
            
            # Check DataFrame
            # The hit at row_idx should have this PID
            actual_pid = tracker_hits_mapped.loc[row_idx, 'particle_id']
            
            if isinstance(actual_pid, pd.Series):
                assert pid in actual_pid.values, \
                    f"Matrix has Hit {row_idx} -> PID {pid}, but DataFrame has PIDs {actual_pid.values}"
            else:
                assert actual_pid == pid, \
                    f"Matrix has Hit {row_idx} -> PID {pid}, but DataFrame has PID {actual_pid}"

    # Calo
    if calo_matrix.nnz > 0:
        rows, cols = calo_matrix.nonzero()
        sample_indices = np.random.choice(len(rows), size=min(n_samples, len(rows)), replace=False)
        
        for i in sample_indices:
            row_idx = rows[i]
            col_idx = cols[i]
            
            pid = unique_pids[col_idx]
            actual_pid = calo_hits_mapped.loc[row_idx, 'contrib_particle_ids']
            
            if isinstance(actual_pid, pd.Series):
                assert pid in actual_pid.values, \
                    f"Matrix has Hit {row_idx} -> PID {pid}, but DataFrame has PIDs {actual_pid.values}"
            else:
                assert actual_pid == pid, \
                    f"Matrix has Hit {row_idx} -> PID {pid}, but DataFrame has PID {actual_pid}"

    print("All spot checks passed!")
