import numpy as np
import pandas as pd
from scipy import sparse

def sparse_association_matrix(particles, tracker_hits, calo_hits):
    """
    Creates sparse look-up matrices (Hits x Particles) for Tracker and Calo.
    Returns scipy.sparse.csr_matrix objects.
    """
    
    # Create the Particle Map (Particle_ID -> 0, 1, 2...)
    unique_particle_ids = particles['particle_id'].unique()
    unique_particle_ids.sort()
    
    # Create a pandas series for fast mapping 
    pid_to_idx_map = pd.Series(
        data=np.arange(len(unique_particle_ids)), 
        index=unique_particle_ids
    )
    
    num_particles = len(unique_particle_ids)
    
    # Function to build vectorized matrix
    def build_matrix_from_df(df, particle_col_name):
        
        # Only keep hits where the particle_id exists in our Truth list
        valid_hits = df[df[particle_col_name].isin(pid_to_idx_map.index)]
        
        # Define Coordinates

        # Row: use df index (which repeats for multi particle hits)
        rows = valid_hits.index.values 
        
        # Col: map the PID to the matrix index using pandas map
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
    tracker_matrix = build_matrix_from_df(tracker_hits, 'particle_id')
    calo_matrix = build_matrix_from_df(calo_hits, 'contrib_particle_ids')

    return tracker_matrix, calo_matrix, unique_particle_ids, pid_to_idx_map