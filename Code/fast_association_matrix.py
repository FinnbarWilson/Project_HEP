import awkward as ak
import numpy as np
import numba as nb
from scipy import sparse
from numba.typed import Dict
from itertools import chain

@nb.njit
def build_id_map(particle_ids):
    """
    Builds a hash map of Particle ID -> Array Index.
    This allows O(1) lookups of particle data by ID.
    """
    id_to_idx = Dict.empty(key_type=nb.int64, value_type=nb.int64)
    for i in range(len(particle_ids)):
        id_to_idx[particle_ids[i]] = i
    return id_to_idx

@nb.njit
def build_ancestry_map_numba(particle_ids, parent_ids, vx, vy, id_to_idx, radius_threshold=2.0):
    """
    Maps every particle to its primary parent (the ancestor produced within radius_threshold).
    Traverses the parent_id chain upwards until a primary parent is found or the chain ends.
    """
    n_particles = len(particle_ids)
    mapped_ids = np.empty(n_particles, dtype=particle_ids.dtype)
    
    for i in range(n_particles):
        curr_id = particle_ids[i]
        curr_idx = i
        
        # Trace ancestry chain upwards
        depth = 0
        while True:
            depth += 1
            if depth > 100:
                # Infinite loop break and assign current as primary
                primary_id = curr_id
                break

            # Check if current particle is a "Parent" (produced near origin)
            r = np.sqrt(vx[curr_idx]**2 + vy[curr_idx]**2)
            if r < radius_threshold:
                primary_id = curr_id
                break
            
            # If not look up its parent
            parent_id = nb.int64(parent_ids[curr_idx])
            
            # Edge Cases: Self-loop or Parent not in event
            if parent_id == curr_id or parent_id not in id_to_idx:
                primary_id = curr_id
                break
                
            # Move up the chain
            curr_id = parent_id
            curr_idx = id_to_idx[curr_id]
            
        # Map current particle to its primary parent
        mapped_ids[i] = primary_id
        
    return mapped_ids

@nb.njit
def fill_hit_matrix(pids_flat, offsets, id_to_idx, mapped_parent_ids, unique_parents):
    """
    Generates (row, col) indices for a CSR matrix.
    Maps hits (Tracker or Calo) to their primary parent's column index.
    Supports 1-to-many associations via flattened arrays and offsets.
    """

    # Calculate the total number of hits based on the offsets array.
    n_hits = len(offsets) - 1

    # Initialize lists to store the row and column indices for the sparse matrix.
    rows = []
    cols = []

    for i in range(n_hits):
        # Determine the start and end indices in the flattened pids_flat array
        # for the current hit i this handles cases where a hit is associated with multiple particles.
        start = offsets[i]
        end = offsets[i+1]
        
        # Iterate through all particle IDs associated with the current hit.
        for j in range(start, end):
            pid = pids_flat[j]
            if pid in id_to_idx:
                p_idx = id_to_idx[pid]
                parent_id = mapped_parent_ids[p_idx]
                
                # Find the column index for primary parent ID in the unique_parents array.
                col_idx = np.searchsorted(unique_parents, parent_id)
                
                if col_idx < len(unique_parents) and unique_parents[col_idx] == parent_id:
                    # If a valid primary parent is found add the hits row index and
                    # the parents column index to the respective lists.
                    rows.append(i)
                    cols.append(col_idx)
    
    return rows, cols


def fast_sparse_association_matrix(events=None, particles=None, tracker_hits=None, calo_hits=None, radius_threshold=2.0):
    """
    Generates sparse association matrices for a batch of events using Numba.
    
    This function processes a batch of events to create sparse matrices linking hits 
    (Tracker and Calorimeter) to their primary parent particles. It efficiently handles 
    ancestry tracing and supports Hugging Face Datasets.

    Args:
        events (dict or ak.Array, optional): A dictionary or Awkward Array containing 
            'particles', 'tracker_hits', and 'calo_hits'.
        particles (Dataset or list, optional): Hugging Face Dataset or list of particle data.
        tracker_hits (Dataset or list, optional): Hugging Face Dataset or list of tracker hit data.
        calo_hits (Dataset or list, optional): Hugging Face Dataset or list of calorimeter hit data.
        radius_threshold (float, default=2.0): The radius (in mm) from the origin within which 
            a particle is considered a "Primary Parent". Particles produced outside this 
            radius are mapped to their ancestor.

    Returns:
        list of tuples: A list where each element corresponds to an event and contains:
            - tracker_matrix (scipy.sparse.csr_matrix): Sparse matrix of shape (n_tracker_hits, n_unique_parents).
            - calo_matrix (scipy.sparse.csr_matrix): Sparse matrix of shape (n_calo_hits, n_unique_parents).
            - unique_parents (np.ndarray): Sorted array of unique primary parent Particle IDs.
            - pid_to_idx (dict): Mapping from Particle ID to the column index in the matrices.
    """
    results = []
    
    # Handle flexible input arguments
    if events is None:
        if particles is None or tracker_hits is None or calo_hits is None:
            raise ValueError("Must provide either 'events' or all of 'particles', 'tracker_hits', 'calo_hits'.")
        
        # Construct events dict from separate args
        events = {
            'particles': particles,
            'tracker_hits': tracker_hits,
            'calo_hits': calo_hits
        }
        num_events = len(particles)
        is_awkward = False
    else:
        if isinstance(events, ak.Array):
            num_events = len(events)
            is_awkward = True
        else:
            num_events = len(events['particles'])
            is_awkward = False
    
    # Pre-fetch Columns
    # Accessing columns once is much faster than accessing rows repeatedly for HF Datasets.
    
    # Particles
    if events is not None and 'particles' in events:
        p_src = events['particles']
    else:
        p_src = particles
    
    p_is_list = isinstance(p_src, list)
    if not p_is_list:
        p_ids_col = p_src['particle_id']
        p_parents_col = p_src['parent_id']
        p_vx_col = p_src['vx']
        p_vy_col = p_src['vy']
    
    # Tracker Hits
    if events is not None and 'tracker_hits' in events:
        t_src = events['tracker_hits']
    else:
        t_src = tracker_hits
        
    t_is_list = isinstance(t_src, list)
    if not t_is_list:
        t_pids_col = t_src['particle_id']
    
    # Calo Hits
    if events is not None and 'calo_hits' in events:
        c_src = events['calo_hits']
    else:
        c_src = calo_hits
        
    c_is_list = isinstance(c_src, list)
    if not c_is_list:
        c_contribs_col = c_src['contrib_particle_ids']

    # Helper to safely convert list with None/NaN to numpy array
    def safe_to_numpy(data, dtype, fill_value=0):
        if isinstance(data, np.ndarray):
            if np.issubdtype(dtype, np.integer) and np.issubdtype(data.dtype, np.floating):
                return np.nan_to_num(data, nan=fill_value).astype(dtype)
            return data.astype(dtype)
        
        if isinstance(data, ak.Array):
            return data.to_numpy().astype(dtype)

        if isinstance(data, list):
            cleaned = [x if x is not None else fill_value for x in data]
            return np.asarray(cleaned, dtype=dtype)
        
        return np.asarray(data, dtype=dtype)
    
    for i in range(num_events):
        # Prepare Data
        
        # Particles
        if p_is_list:
            p_data = p_src[i]
            p_ids = safe_to_numpy(p_data['particle_id'], np.int64)
            p_parents = safe_to_numpy(p_data['parent_id'], np.int64)
            p_vx = safe_to_numpy(p_data['vx'], np.float64)
            p_vy = safe_to_numpy(p_data['vy'], np.float64)
        else:
            p_ids = safe_to_numpy(p_ids_col[i], np.int64)
            p_parents = safe_to_numpy(p_parents_col[i], np.int64)
            p_vx = safe_to_numpy(p_vx_col[i], np.float64)
            p_vy = safe_to_numpy(p_vy_col[i], np.float64)
        
        # Build Maps
        id_to_idx = build_id_map(p_ids)
        mapped_parent_ids = build_ancestry_map_numba(p_ids, p_parents, p_vx, p_vy, id_to_idx, radius_threshold)
        
        # Identify unique parents (Columns of the matrix)
        unique_parents = np.unique(mapped_parent_ids)
        pid_to_idx = {pid: idx for idx, pid in enumerate(unique_parents)}
        num_columns = len(unique_parents)
        
        # Build Tracker Matrix
        if t_is_list:
            t_data = t_src[i]
            t_raw = t_data['particle_id']
        else:
            t_raw = t_pids_col[i]
            
        # Handle both 1-to-1 (common) and 1-to-many (rare) tracker hits
        is_nested = False
        if len(t_raw) > 0:
            # Check first element to see if it's a list/array (1-to-many)
            first_item = t_raw[0]
            if isinstance(first_item, (list, np.ndarray, ak.Array)):
                is_nested = True
            elif isinstance(t_raw, ak.Array) and t_raw.ndim > 1:
                is_nested = True
        
        if is_nested:
            # Flatten list-of-lists
            if isinstance(t_raw, ak.Array):
                t_pids_flat = ak.flatten(t_raw, axis=None).to_numpy()
                t_counts = ak.to_numpy(ak.num(t_raw, axis=0))
            else:
                t_pids_flat = np.fromiter(chain.from_iterable(t_raw), dtype=np.int64)
                t_counts = np.array([len(x) for x in t_raw], dtype=np.int64)
            t_offsets = np.concatenate(([0], np.cumsum(t_counts)))
        else:
            # 1-to-1 case
            t_pids_flat = safe_to_numpy(t_raw, np.int64)
            t_offsets = np.arange(len(t_pids_flat) + 1, dtype=np.int64)
            
        t_rows, t_cols = fill_hit_matrix(t_pids_flat, t_offsets, id_to_idx, mapped_parent_ids, unique_parents)
        
        num_t_hits = len(t_raw)
        tracker_matrix = sparse.coo_matrix(
            (np.ones(len(t_rows)),
            (t_rows, t_cols)),
            shape=(num_t_hits, num_columns)
        ).tocsr()
        
        # Build Calo Matrix
        if c_is_list:
            c_data = c_src[i]
            c_contribs = c_data['contrib_particle_ids']
        else:
            c_contribs = c_contribs_col[i]
        
        # Flatten list-of-lists efficiently
        if is_awkward:
            c_pids_flat = ak.flatten(c_contribs, axis=None).to_numpy()
            c_counts = ak.to_numpy(ak.num(c_contribs, axis=0))
        else:
            c_pids_flat = np.fromiter(chain.from_iterable(c_contribs), dtype=np.int64)
            c_counts = np.array([len(x) for x in c_contribs], dtype=np.int64)
            
        c_offsets = np.concatenate(([0], np.cumsum(c_counts)))
        
        c_rows, c_cols = fill_hit_matrix(c_pids_flat, c_offsets, id_to_idx, mapped_parent_ids, unique_parents)
        
        num_c_hits = len(c_contribs)
        calo_matrix = sparse.coo_matrix(
            (np.ones(len(c_rows)),
            (c_rows, c_cols)),
            shape=(num_c_hits, num_columns)
        ).tocsr()
        
        results.append((tracker_matrix, calo_matrix, unique_parents, pid_to_idx))
        
    return results
