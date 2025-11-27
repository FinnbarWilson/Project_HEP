import numpy as np
import pandas as pd
from scipy import sparse

def map_daughters_to_parents(particles, hits, particle_col='particle_id', radius_threshold=2.0):
    """
    Maps hits from daughter particles to their primary parent particle.
    
    A "Parent" is defined as a particle produced within `radius_threshold` from the origin.
    Any particle produced outside this radius is considered a "Daughter".
    Hits from daughters are reassigned to their ultimate "Primary Parent".
    """
    
    # Identify Parents vs Daughters
    r = np.sqrt(particles['vx']**2 + particles['vy']**2)
    is_parent_mask = r < radius_threshold
    parent_ids = set(particles.loc[is_parent_mask, 'particle_id'])
    
    # Dictionary for fast lookups: pid -> parent_pid
    pid_to_parent = dict(zip(particles['particle_id'], particles['parent_id']))
    
    # Build Ancestry Map (Daughter -> Ultimate Parent)
    ancestry_map = {}
    
    def get_primary_parent(pid):
        """Iteratively finds the primary parent for a given particle ID."""
        path = []
        curr = pid
        
        while True:
            # Check 1: Already solved?
            if curr in ancestry_map:
                primary = ancestry_map[curr]
                break
            
            # Check 2: Is this a "Parent" or unknown root?
            if curr in parent_ids or curr not in pid_to_parent:
                primary = curr
                break
            
            # It's a daughter, move up the chain
            path.append(curr)
            parent = pid_to_parent[curr]
            
            # Check 3: Cycle detection or self-loop
            if parent == curr or parent in path:
                primary = curr 
                break
                
            curr = parent
            
        # Update map for all visited particles
        for p in path:
            ancestry_map[p] = primary
        ancestry_map[curr] = primary
        
        return primary

    # Pre-compute map for all particles
    for pid in particles['particle_id']:
        get_primary_parent(pid)
        
    # Update Hits
    modified_hits = hits.copy()
    
    # Map IDs, keeping original if not found in map (eg noise)
    modified_hits[particle_col] = modified_hits[particle_col].map(ancestry_map).fillna(modified_hits[particle_col]).infer_objects(copy=False)

    return modified_hits

def _process_single_event(particles, tracker_hits, calo_hits, map_daughters=True, radius_threshold=2.0):
    """
    Internal function to process a single event (Pandas DataFrames).
    """
    
    # Pre-process hits to map daughters to parents
    if map_daughters:
        tracker_hits_mapped = map_daughters_to_parents(
            particles, tracker_hits, 'particle_id', radius_threshold
        )
        
        calo_hits_mapped = map_daughters_to_parents(
            particles, calo_hits, 'contrib_particle_ids', radius_threshold
        )
    else:
        tracker_hits_mapped = tracker_hits
        calo_hits_mapped = calo_hits
    
    # Create Particle Map (Particle_ID -> Matrix Column Index)
    unique_particle_ids = particles['particle_id'].unique()
    unique_particle_ids.sort()
    
    pid_to_idx_map = pd.Series(
        data=np.arange(len(unique_particle_ids)), 
        index=unique_particle_ids
    )
    
    num_particles = len(unique_particle_ids)
    
    def build_matrix_from_df(df, particle_col_name):
        """Builds a sparse matrix from a hits DataFrame."""
        
        # Filter hits associated with known particles
        valid_hits = df[df[particle_col_name].isin(pid_to_idx_map.index)]
        
        # Define Coordinates for Sparse Matrix
        # Row: df index (repeats for multi-particle hits)
        rows = valid_hits.index.values 
        
        # Col: mapped matrix index
        cols = valid_hits[particle_col_name].map(pid_to_idx_map).values
        
        # Data: list of 1s
        data = np.ones(len(rows))
        
        # Matrix Shape: (Max Hit ID + 1) x (Num Particles)
        num_hits = df.index.max() + 1 
        
        matrix = sparse.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_hits, num_particles)
        ).tocsr()
        
        return matrix

    # Build Matrices
    tracker_matrix = build_matrix_from_df(tracker_hits_mapped, 'particle_id')
    calo_matrix = build_matrix_from_df(calo_hits_mapped, 'contrib_particle_ids')

    return tracker_matrix, calo_matrix, unique_particle_ids, pid_to_idx_map

def sparse_association_matrix(events=None, particles=None, tracker_hits=None, calo_hits=None, radius_threshold=2.0):
    """
    Generates sparse association matrices for a batch of events.
    """
    results = []
    
    # Handle flexible input arguments
    if events is None:
        if particles is None or tracker_hits is None or calo_hits is None:
            raise ValueError("Must provide either 'events' or all of 'particles', 'tracker_hits', 'calo_hits'.")
        # Use separate datasets
        p_src = particles
        t_src = tracker_hits
        c_src = calo_hits
        num_events = len(particles)
    else:
        # Use events dict/array
        p_src = events['particles']
        t_src = events['tracker_hits']
        c_src = events['calo_hits']
        if isinstance(events, dict):
            num_events = len(events['particles'])
        else:
            num_events = len(events)
            
    for i in range(num_events):
        # Extract data for single event
        
        # Particles
        if isinstance(p_src, list): 
            p_data = p_src[i]
        else: 
            # HF Dataset or Dict of arrays
            p_cols = ['particle_id', 'parent_id', 'vx', 'vy']
            p_data = {k: p_src[k][i] for k in p_cols}
        
        # Tracker
        if isinstance(t_src, list): 
            t_data = t_src[i]
        else: 
            t_cols = ['particle_id']
            t_data = {k: t_src[k][i] for k in t_cols}
        
        # Calo
        if isinstance(c_src, list): 
            c_data = c_src[i]
        else: 
            c_cols = ['contrib_particle_ids']
            c_data = {k: c_src[k][i] for k in c_cols}
        
        # Convert to DataFrames (Exploded)
        df_p = pd.DataFrame(p_data)
        df_t = pd.DataFrame(t_data)
        df_c = pd.DataFrame(c_data)
        df_c = df_c.explode('contrib_particle_ids')
        
        # Process Single Event
        try:
            res = _process_single_event(df_p, df_t, df_c, radius_threshold=radius_threshold)
            results.append(res)
        except Exception as e:
            # Skip failing events (eg duplicate indices)
            results.append(None)
            pass
    
    return results