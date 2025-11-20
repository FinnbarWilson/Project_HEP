# 3D visulisation of tracks

[Can be viewed here](https://finnbarwilson.github.io/Project_HEP/Code/Interactive3Dviewer.html)

# .Parquet Files:

Can load data into pandas using `get_HEP_data.py`. 

**Important Notes:**
1.  **Exploded Columns**: The function automatically `explodes` list columns (like `contrib_particle_ids` in Calo hits) into scalar values. This means a single physical hit might appear as multiple rows if it has multiple contributing particles.
2.  **HIT_ID Index**: For `calo_hits` and `tracker_hits`, the DataFrame index is set to `HIT_ID`. This ID is unique to the physical hit. If a hit is exploded into multiple rows, they will all share the same `HIT_ID` index.

Below is the head of each file:

### particles

| **event_id** | **particle_id** | **pdg_id** | **mass** | **energy** | **charge** | **vx** | **vy** | **vz** | **time** | **px** | **py** | **pz** | **num_tracker_hits** | **num_calo_hits** | **vertex_primary** | **parent_id** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 76 | 213 | 0.738 | 2491.61 | 1.0 | 0.009 | -0.004 | 196.07 | 11.96 | 0.033 | -0.117 | 2491.61 | 0 | 0 | 1 | 9.0 |
| 0 | 77 | -211 | 0.140 | 489.57 | -1.0 | 0.009 | -0.004 | 196.07 | 11.96 | -0.343 | 0.318 | 489.57 | 0 | 0 | 1 | 9.0 |
| 0 | 78 | 211 | 0.140 | 199.21 | 1.0 | 0.009 | -0.004 | 196.07 | 11.96 | 0.114 | -0.173 | 199.21 | 0 | 0 | 1 | 9.0 |
| 0 | 79 | 113 | 0.702 | 140.08 | 0.0 | 0.009 | -0.004 | 196.07 | 11.96 | 0.344 | 0.453 | 140.07 | 0 | 0 | 1 | 9.0 |
| 0 | 80 | 111 | 0.135 | 50.87 | 0.0 | 0.009 | -0.004 | 196.07 | 11.96 | -0.084 | -0.565 | 50.87 | 0 | 0 | 1 | 9.0 |

### calo_hits

(Note that `get_HEP_data.py` takes the values out of the arrays in each contrib_ column) 

*Index is `HIT_ID`.*

| **HIT_ID** | **event_id** | **detector** | **cell_id** | **total_energy** | **x** | **y** | **z** | **contrib_particle_ids** | **contrib_energies** | **contrib_times** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 1 | ECalBarrel... | 18430... | 0.000448 | -545.24 | 1129.74 | -300.90 | 361 | 0.000270 | 8.479 |
| **0** | 1 | ECalBarrel... | 18430... | 0.000448 | -545.24 | 1129.74 | -300.90 | 383 | 0.000179 | 8.473 |
| **1** | 1 | ECalBarrel... | 18430... | 0.000504 | -547.17 | 1134.41 | -300.90 | 383 | 0.000504 | 8.491 |
| **2** | 1 | ECalBarrel... | 18430... | 0.001057 | -549.10 | 1139.07 | -300.90 | 383 | 0.001057 | 8.511 |
| **3** | 1 | ECalBarrel... | 18429... | 0.000455 | -551.04 | 1143.74 | -306.00 | 383 | 0.000455 | 8.527 |

### tracker_hits

*Index is `HIT_ID`.*

| **HIT_ID** | **event_id** | **x** | **y** | **z** | **time** | **particle_id** | **true_x** | **true_y** | **true_z** | **volume_id** | **layer_id** | **surface_id** | **cell_id** | **detector** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | 0 | 41.84 | 7.77 | -1516.80 | 17.67 | 1787 | 41.83 | 7.78 | -1516.80 | 16 | 4 | 2 | 2834... | 1 |
| **1** | 0 | -81.95 | -37.41 | -1515.60 | 17.68 | 276 | -81.93 | -37.43 | -1515.60 | 16 | 4 | 7 | 7344... | 1 |
| **2** | 0 | 66.75 | -67.07 | -1516.80 | 17.68 | 280 | 66.75 | -67.08 | -1516.80 | 16 | 4 | 15 | 2147... | 1 |
| **3** | 0 | -0.08 | 45.53 | -1515.60 | 17.67 | 332 | -0.07 | 45.55 | -1515.60 | 16 | 4 | 21 | 1758... | 1 |
| **4** | 0 | 2.73 | 99.47 | -1515.60 | 17.69 | 281 | 2.70 | 99.48 | -1515.60 | 16 | 4 | 21 | 2319... | 1 |

### tracks

(Note that `get_HEP_data.py` takes the values out of the array in hit_ids column)

| **event_id** | **majority_particle_id** | **d0** | **z0** | **phi** | **theta** | **qop** | **hit_ids** | **track_id** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 301 | 0.024 | 196.07 | 3.06 | 2.26 | 0.76 | 141 | 16 |
| 0 | 301 | 0.024 | 196.07 | 3.06 | 2.26 | 0.76 | 165 | 16 |
| 0 | 301 | 0.024 | 196.07 | 3.06 | 2.26 | 0.76 | 168 | 16 |
| 0 | 301 | 0.024 | 196.07 | 3.06 | 2.26 | 0.76 | 301 | 16 |
| 0 | 301 | 0.024 | 196.07 | 3.06 | 2.26 | 0.76 | 403 | 16 |