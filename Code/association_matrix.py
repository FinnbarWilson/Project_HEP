import numpy as np
import pandas as pd
from scipy import sparse

def map_daughters_to_parents(particles, hits, particle_col='particle_id', radius_threshold=2.0):
    """
    Maps hits from daughter particles to their primary parent particle.
    
    A "Parent" is defined as a particle produced within `radius_threshold` from the origin (sqrt(vx^2 + vy^2) < threshold).
    Any particle produced outside this radius is considered a "Daughter".
    
    For each daughter particle, we trace its ancestry using `parent_id` until we find a Parent.
    The hits associated with the daughter are then reassigned to that Parent.
    
    Args:
        particles: DataFrame of particles with 'particle_id', 'parent_id', 'vx', 'vy'.
        hits: DataFrame of hits with a column containing particle IDs.
        particle_col: Name of the column in `hits` containing particle IDs.
        radius_threshold: Radius in mm (or same units as vx, vy) to define the production vertex threshold.
        
    Returns:
        modified_hits: A copy of the hits DataFrame with updated particle IDs.
    """
    
    # 1. Identify Parents vs Daughters
    # Calculate production radius
    r = np.sqrt(particles['vx']**2 + particles['vy']**2)
    
    # Create a mapping: particle_id -> is_parent (boolean)
    is_parent_mask = r < radius_threshold
    parent_ids = set(particles.loc[is_parent_mask, 'particle_id'])
    
    # Create a dictionary for fast lookups: pid -> parent_pid
    pid_to_parent = dict(zip(particles['particle_id'], particles['parent_id']))
    
    # 2. Build Ancestry Map (Daughter -> Ultimate Parent)
    # We want to map every particle ID to its "Primary Parent" ID.
    # If a particle is already a Parent, it maps to itself.
    
    ancestry_map = {}
    
    def get_primary_parent(pid):
        """
        Iteratively finds the primary parent for a given particle ID.
        """
        path = []
        curr = pid
        
        while True:
            # Check 1: Have we already solved this particle?
            if curr in ancestry_map:
                # We found a known ancestor, all in path map to this
                primary = ancestry_map[curr]
                break
            
            # Check 2: Is this particle a "Parent" (produced near origin)?
            # Or is it unknown (not in our list)? In either case, it's the root.
            if curr in parent_ids or curr not in pid_to_parent:
                primary = curr
                break
            
            # It's a daughter, move up the chain
            path.append(curr)
            parent = pid_to_parent[curr]
            
            # Check 3: Cycle detection or self-loop
            if parent == curr or parent in path:
                # Cycle detected or self-loop. Break and assign to self/current parent.
                primary = curr 
                break
                
            curr = parent
            
        # Update the map for every particle we visited on the way up.
        # Next time we ask about any of them, we'll get the answer instantly.
        for p in path:
            ancestry_map[p] = primary
        ancestry_map[curr] = primary # Ensure the root is also mapped
        
        return primary

    # Pre-compute map for all particles involved in hits
    for pid in particles['particle_id']:
        get_primary_parent(pid)
        
    # 3. Update Hits
    modified_hits = hits.copy()
    
    # Use map to replace values. 
    # We need to handle the case where hits contain IDs not in our ancestry_map (e.g. noise)
    # We keep them as is (fillna).
    # Since get_HEP_data.py explodes the columns, we can assume scalar values here.
    modified_hits[particle_col] = modified_hits[particle_col].map(ancestry_map).fillna(modified_hits[particle_col])

    return modified_hits

def sparse_association_matrix(particles, tracker_hits, calo_hits, map_daughters=True, radius_threshold=2.0):
    """
    Creates sparse look-up matrices (Hits x Particles) for Tracker and Calo.
    
    The output is a Compressed Sparse Row (CSR) matrix where:
    - Rows correspond to Hits (indexed by hit_id)
    - Columns correspond to Particles (mapped from particle_id to 0..N)
    - A value of 1 indicates the hit is associated with that particle.
    
    Args:
        particles: DataFrame containing particle truth information.
        tracker_hits: DataFrame containing tracker hit information.
        calo_hits: DataFrame containing calorimeter hit information.
        map_daughters: If True, maps daughter particles to their parents.
        radius_threshold: Radius threshold for defining parent particles (default 2.0).

    Returns:
        tracker_matrix (scipy.sparse.csr_matrix): Hits x Particles matrix for Tracker.
        calo_matrix (scipy.sparse.csr_matrix): Hits x Particles matrix for Calo.
        unique_particle_ids (array): Array of sorted unique particle IDs found in the truth file.
        pid_to_idx_map (pd.Series): Mapping from Real Particle ID -> Matrix Column Index.
    """
    
    # Pre-process hits to map daughters to parents if requested
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
    
    # Create the Particle Map (Particle_ID -> 0, 1, 2...)
    
    unique_particle_ids = particles['particle_id'].unique()
    unique_particle_ids.sort()
    
    # Create a pandas series for fast mapping: Real ID -> Matrix Column Index
    # Example: Particle IDs [10, 55, 100] -> Column Indices [0, 1, 2]
    pid_to_idx_map = pd.Series(
        data=np.arange(len(unique_particle_ids)), 
        index=unique_particle_ids
    )
    
    num_particles = len(unique_particle_ids)
    
    # Function to build vectorized matrix
    def build_matrix_from_df(df, particle_col_name):
        
        # Only keep hits where the particle_id exists (associated with known particles)
        valid_hits = df[df[particle_col_name].isin(pid_to_idx_map.index)]
        
        # Define Coordinates for the Sparse Matrix
        # A sparse matrix is built from a list of (row, col) coordinates where values exist.

        # Row: use df index (which repeats for multi particle hits)
        # Example: If hit 10 has 2 particles, we have two entries with row=10.
        rows = valid_hits.index.values 
        
        # Col: map the PID to the matrix index using pandas map
        # Example: Particle ID 55 -> Column 1
        cols = valid_hits[particle_col_name].map(pid_to_idx_map).values
        
        # Data: Just Ones
        data = np.ones(len(rows))
        
        # Define Matrix Shape
        # The height of the matrix is the largest Index ID + 1
        num_hits = df.index.max() + 1 
        
        # Build Sparse Matrix
        matrix = sparse.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_hits, num_particles)
        ).tocsr()
        
        return matrix

    # Build the Matrices
    tracker_matrix = build_matrix_from_df(tracker_hits_mapped, 'particle_id')
    calo_matrix = build_matrix_from_df(calo_hits_mapped, 'contrib_particle_ids')

    return tracker_matrix, calo_matrix, unique_particle_ids, pid_to_idx_map