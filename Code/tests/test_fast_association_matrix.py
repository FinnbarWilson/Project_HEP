import pandas as pd
import numpy as np
import os
import sys
from scipy import sparse

# Add Code directory to path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fast_association_matrix import fast_sparse_association_matrix, build_ancestry_map_numba

# Define paths to local data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data/parquet')

PARTICLES_PATH = os.path.join(DATA_DIR, 'truth/particles/hard_scatter.ttbar.v1.truth.particles.events0-9.parquet')
TRACKER_PATH = os.path.join(DATA_DIR, 'reco/tracker_hits/hard_scatter.ttbar.v1.reco.tracker_hits.events0-9.parquet')
CALO_PATH = os.path.join(DATA_DIR, 'reco/calo_hits/hard_scatter.ttbar.v1.reco.calo_hits.events0-9.parquet')

def load_local_data_as_batch(num_events=10):
    """
    Loads local parquet files and structures them as a batch dictionary
    mimicking the Hugging Face dataset structure.
    """
    # Load DataFrames
    df_particles = pd.read_parquet(PARTICLES_PATH)
    df_tracker = pd.read_parquet(TRACKER_PATH)
    df_calo = pd.read_parquet(CALO_PATH)
    
    # Get list of event IDs
    event_ids = sorted(df_particles['event_id'].unique())[:num_events]
    
    batch = {
        'particles': [],
        'tracker_hits': [],
        'calo_hits': []
    }
    
    for eid in event_ids:
        # Filter for this event
        p_evt = df_particles[df_particles['event_id'] == eid]
        t_evt = df_tracker[df_tracker['event_id'] == eid]
        c_evt = df_calo[df_calo['event_id'] == eid]
        
        print(f"Event {eid}: Particles={len(p_evt)}, Tracker={len(t_evt)}, Calo={len(c_evt)}")
        
        if len(p_evt) == 0 or len(t_evt) == 0 or len(c_evt) == 0:
            print(f"Skipping Event {eid} due to missing data")
            continue

        batch['particles'].append(p_evt.iloc[0].to_dict())
        batch['tracker_hits'].append(t_evt.iloc[0].to_dict())
        batch['calo_hits'].append(c_evt.iloc[0].to_dict())
        
    return batch

def test_fast_sparse_association_matrix():
    """
    Test that the fast association matrix generation works on local data.
    """
    # Load Data
    try:
        batch = load_local_data_as_batch(num_events=2)
    except FileNotFoundError as e:
        pytest.skip(f"Local data not found: {e}")
        
    # Run Pipeline
    results = fast_sparse_association_matrix(batch, radius_threshold=2.0)
    
    # Verify Output
    assert len(results) > 0, "Should return results for at least one event"
    
    for i, (t_mat, c_mat, pids, pid_map) in enumerate(results):
        print(f"Verifying Event {i}...")
        
        # Check Types
        assert sparse.issparse(t_mat), "Tracker matrix should be sparse"
        assert sparse.issparse(c_mat), "Calo matrix should be sparse"
        assert isinstance(pids, (np.ndarray, list)), "Unique PIDs should be array/list"
        assert isinstance(pid_map, dict), "PID map should be a dictionary"
        
        # Check Shapes
        # Number of columns should equal number of unique parents
        assert t_mat.shape[1] == len(pids)
        assert c_mat.shape[1] == len(pids)
        
        # Check Content (Sanity)
        # We expect some hits to be associated
        assert t_mat.nnz > 0, "Tracker matrix should have non-zero entries"
        # Calo might be empty if no hits or threshold issues, but usually has entries
        assert c_mat.nnz >= 0 
        
        print(f"Event {i} Passed: T={t_mat.shape}, C={c_mat.shape}, P={len(pids)}")
        
        # Validation: Check Hit Counts
        print("  Validating hit counts...")
        
        # Reconstruct Parent Map for Validation
        # We use the same Numba function to get the ground truth mapping
        p_data = batch['particles'][i]
        p_ids = np.array(p_data['particle_id'])
        p_parents = np.array(p_data['parent_id'])
        p_vx = np.array(p_data['vx'])
        p_vy = np.array(p_data['vy'])
        
        from fast_association_matrix import build_id_map
        id_to_idx = build_id_map(p_ids)
        mapped_parents = build_ancestry_map_numba(p_ids, p_parents, p_vx, p_vy, id_to_idx, radius_threshold=2.0)
        
        # Map: Original ID -> Mapped Parent ID
        orig_to_parent = dict(zip(p_ids, mapped_parents))
        
        # Map: Mapped Parent ID -> Matrix Column Index
        # (This is what the function returns as pid_map)
        parent_to_col = pid_map
        
        # Validate Tracker Hits
        t_hits = batch['tracker_hits'][i]
        t_pids = t_hits['particle_id']
        
        # Calculate expected counts
        expected_t_counts = np.zeros(len(pids))
        
        for pid in t_pids:
            if pid in orig_to_parent:
                parent = orig_to_parent[pid]
                if parent in parent_to_col:
                    col_idx = parent_to_col[parent]
                    expected_t_counts[col_idx] += 1
                    
        # Compare with Matrix Column Sums
        # axis=0 sums over rows (hits), giving total hits per particle (column)
        actual_t_counts = np.array(t_mat.sum(axis=0)).flatten()
        
        np.testing.assert_array_equal(actual_t_counts, expected_t_counts, err_msg="Tracker hit counts do not match!")
        print("  Tracker hits validated.")
        
        # Validate Calo Hits
        c_hits = batch['calo_hits'][i]
        c_contribs = c_hits['contrib_particle_ids']
        
        expected_c_counts = np.zeros(len(pids))
        
        for contrib_list in c_contribs:
            if contrib_list is None: continue
            for pid in contrib_list:
                if pid in orig_to_parent:
                    parent = orig_to_parent[pid]
                    if parent in parent_to_col:
                        col_idx = parent_to_col[parent]
                        expected_c_counts[col_idx] += 1
                        
        actual_c_counts = np.array(c_mat.sum(axis=0)).flatten()
        
        np.testing.assert_array_equal(actual_c_counts, expected_c_counts, err_msg="Calo hit counts do not match!")
        print("  Calo hits validated.")
        
        # Validation: Round-Trip (Matrix -> Dataset)
        print("  Validating round-trip (Matrix -> Dataset)...")
        
        # Invert pid_map: Column Index -> Parent ID
        col_to_parent = {v: k for k, v in pid_map.items()}
        
        # Verify Tracker Matrix Associations
        # Iterate over all non-zero entries (associations)
        t_coo = t_mat.tocoo()
        for hit_idx, col_idx in zip(t_coo.row, t_coo.col):
            # The matrix says: hit_idx belongs to parent_id
            parent_id = col_to_parent[col_idx]
            
            # Check the original data
            original_pid = t_hits['particle_id'][hit_idx]
            
            # The original particle must map to this parent
            expected_parent = orig_to_parent.get(original_pid)
            
            assert expected_parent == parent_id, \
                f"Tracker Mismatch! Hit {hit_idx} (PID {original_pid}) maps to {expected_parent}, but matrix has {parent_id}"
                
        print("  Tracker round-trip validated.")
        
        # Verify Calo Matrix Associations
        c_coo = c_mat.tocoo()
        for hit_idx, col_idx in zip(c_coo.row, c_coo.col):
            parent_id = col_to_parent[col_idx]
            
            # Check original data
            contrib_list = c_hits['contrib_particle_ids'][hit_idx]
            
            # One of the contributing particles must map to this parent
            found_match = False
            for original_pid in contrib_list:
                if orig_to_parent.get(original_pid) == parent_id:
                    found_match = True
                    break
            
            assert found_match, \
                f"Calo Mismatch! Hit {hit_idx} (Contribs {contrib_list}) does not contain parent {parent_id}"
                
        print("  Calo round-trip validated.")
