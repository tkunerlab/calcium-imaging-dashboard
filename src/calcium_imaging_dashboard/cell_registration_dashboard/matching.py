import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def summarize_matching(matching_matrix):
    """Return one canonical set of track counts and multi-session coverage."""
    matrix = np.asarray(matching_matrix, dtype=float)
    total = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    if total == 0:
        return {
            "n_master_cells": 0,
            "n_matched_cells": 0,
            "n_unmatched_cells": 0,
            "coverage_pct": 0.0,
        }
    presence = np.sum(~np.isnan(matrix), axis=1)
    matched = int(np.sum(presence >= 2))
    unmatched = int(np.sum(presence == 1))
    return {
        "n_master_cells": total,
        "n_matched_cells": matched,
        "n_unmatched_cells": unmatched,
        "coverage_pct": 100.0 * matched / total,
    }


def delete_master_cell_groups(
    matching_matrix, master_centroids, master_footprints, master_indices
):
    """Remove master rows and remap per-session local cell indices.

    Returns the reduced matching data plus the local cell indices that must be
    discarded from each session column.
    """
    matrix = np.asarray(matching_matrix, dtype=float)
    centroids = np.asarray(master_centroids, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Matching matrix must be two-dimensional.")
    selected = sorted({int(index) for index in master_indices})
    if not selected:
        raise ValueError("Select at least one master cell group to delete.")
    if selected[0] < 0 or selected[-1] >= matrix.shape[0]:
        raise IndexError("Selected master cell group is outside the matching matrix.")

    deleted_local_indices = []
    for column in range(matrix.shape[1]):
        values = matrix[selected, column]
        deleted_local_indices.append(
            sorted({int(value) for value in values if not np.isnan(value) and value >= 0})
        )

    keep_mask = np.ones(matrix.shape[0], dtype=bool)
    keep_mask[selected] = False
    reduced = matrix[keep_mask, :].copy()
    reduced_centroids = centroids[keep_mask, :].copy()
    footprints = list(master_footprints) if master_footprints is not None else []
    reduced_footprints = (
        [footprints[index] for index in range(len(footprints)) if keep_mask[index]]
        if len(footprints) == matrix.shape[0] else []
    )

    for column, deleted in enumerate(deleted_local_indices):
        if not deleted:
            continue
        for row in range(reduced.shape[0]):
            value = reduced[row, column]
            if np.isnan(value) or value < 0:
                continue
            local_index = int(value)
            if local_index in deleted:
                reduced[row, column] = np.nan
            else:
                reduced[row, column] = local_index - int(
                    np.searchsorted(deleted, local_index, side="left")
                )

    populated = np.any(~np.isnan(reduced), axis=1)
    if not np.all(populated):
        reduced = reduced[populated, :]
        reduced_centroids = reduced_centroids[populated, :]
        if reduced_footprints:
            reduced_footprints = [
                footprint for footprint, keep in zip(reduced_footprints, populated) if keep
            ]
    return reduced, reduced_centroids, reduced_footprints, deleted_local_indices

def get_sparse_footprints(spatial, threshold_fraction=0.01):
    """Extracts active pixels to create sparse footprint dictionaries for fast overlap calculation.
    
    Equivalent to functions/matchCellsAcrossSessions.m -> getSparseFootprints
    """
    N, H, W = spatial.shape
    sparse_footprints = []
    
    for i in range(N):
        footprint = spatial[i]
        max_val = np.max(footprint)
        thr = threshold_fraction * max_val
        if thr <= 0:
            thr = 0.01
            
        # Flat indices in column-major order (order='F') matching MATLAB
        flat_footprint = footprint.ravel(order='F')
        idx_0 = np.where(flat_footprint > thr)[0]
        vals = flat_footprint[idx_0]
        
        # Convert to 1-based indexing for MATLAB compatibility
        idx_1 = (idx_0 + 1).astype('uint32')
        
        sparse_footprints.append({
            'idx': idx_1,
            'vals': vals.astype('float64'),
            'norm': float(np.linalg.norm(vals))
        })
        
    return sparse_footprints

def shift_sparse_footprint(sf, dx, dy, H, W):
    """Shifts coordinate indices of a sparse footprint by dx and dy, matching MATLAB's shiftSparseFootprint."""
    if len(sf['idx']) == 0:
        return sf
    
    # ind2sub: Convert 1-based flat index to 1-based row and column (column-major)
    idx_0 = sf['idx'] - 1
    r = idx_0 % H + 1
    c = idx_0 // H + 1
    
    # Apply shift
    r_new = np.round(r + dy).astype(int)
    c_new = np.round(c + dx).astype(int)
    
    # Boundary checks (1-based coordinate bounds)
    valid = (r_new >= 1) & (r_new <= H) & (c_new >= 1) & (c_new <= W)
    if not np.any(valid):
        return {'idx': np.empty((0,), dtype='uint32'), 'vals': np.empty((0,), dtype='float64'), 'norm': 1.0}
    
    # sub2ind: Convert row/col back to 1-based flat index (column-major)
    idx_new_0 = (c_new[valid] - 1) * H + (r_new[valid] - 1)
    idx_new_1 = (idx_new_0 + 1).astype('uint32')
    vals_new = sf['vals'][valid]
    
    # Sort indices to match MATLAB's intersect/union requirements
    sort_order = np.argsort(idx_new_1)
    idx_new_1 = idx_new_1[sort_order]
    vals_new = vals_new[sort_order]
    
    return {
        'idx': idx_new_1,
        'vals': vals_new,
        'norm': float(np.linalg.norm(vals_new))
    }

def compute_overlap(sf_i, sf_j, overlap_type='cosine'):
    """Computes spatial overlap (Cosine or Jaccard IoU) between two sparse footprints."""
    idx_i = sf_i['idx']
    idx_j = sf_j['idx']
    
    if len(idx_i) == 0 or len(idx_j) == 0:
        return 0.0
    
    if overlap_type == 'jaccard':
        intersect_len = len(np.intersect1d(idx_i, idx_j, assume_unique=True))
        union_len = len(np.union1d(idx_i, idx_j))
        if union_len > 0:
            return float(intersect_len) / float(union_len)
        return 0.0
    else: # cosine
        intersect_vals, idx_in_i, idx_in_j = np.intersect1d(idx_i, idx_j, assume_unique=True, return_indices=True)
        if len(idx_in_i) == 0:
            return 0.0
        
        numerator = np.sum(sf_i['vals'][idx_in_i] * sf_j['vals'][idx_in_j])
        den = sf_i['norm'] * sf_j['norm']
        if den > 0:
            return float(numerator / den)
        return 0.0

def solve_matching_pairs(cost_matrix, max_cost=1.0):
    """Solves minimum cost linear assignment (Hungarian), filtering matches exceeding max_cost.
    
    Equivalent to MATLAB's matchpairs(cost_matrix, 1.0).
    """
    cost_matrix = np.copy(cost_matrix)
    
    # Replace Inf/NaN with a large number
    cost_matrix[np.isinf(cost_matrix)] = 9999.0
    cost_matrix[np.isnan(cost_matrix)] = 9999.0
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    valid_pairs = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < max_cost:
            valid_pairs.append((r, c))
            
    return np.array(valid_pairs, dtype=int) if valid_pairs else np.empty((0, 2), dtype=int)

def match_cells_across_sessions(sessions_list, max_dist=20.0, min_overlap=0.03, cost_weight=0.50, overlap_type='cosine'):
    """Matches cells across multiple sessions chronologically, updating a master union map.
    
    Parameters:
        sessions_list: List of dicts, each containing:
            'session_name': str
            'n_cells': int
            'centroids': numpy array (N x 2)
            'sparse_footprints': list of sparse footprint dicts
        
    Returns:
        matching_matrix: M x S numpy array of matched cell indices (0-indexed, NaNs for missing matches)
        master_centroids: M x 2 array of master centroids
        master_footprints: list of M master sparse footprints
    """
    S = len(sessions_list)
    if S == 0:
        return np.empty((0, 0)), np.empty((0, 2)), []
        
    # Find reference session (session with most cells)
    cell_counts = [sess['n_cells'] for sess in sessions_list]
    ref_idx = int(np.argmax(cell_counts))
    ref_sess = sessions_list[ref_idx]
    
    # Reorder sessions list so reference is first
    ordered_idx = [ref_idx] + [i for i in range(S) if i != ref_idx]
    
    # Initialize Master Union Map with reference session cells
    master_centroids = np.copy(ref_sess['centroids'])
    master_footprints = list(ref_sess['sparse_footprints'])
    
    # matching_matrix: rows = Master Cell ID, cols = Sessions. Reordered order first.
    matching_matrix = np.arange(ref_sess['n_cells'], dtype=float).reshape(-1, 1)
    
    for s_order_idx in range(1, S):
        curr_sess_idx = ordered_idx[s_order_idx]
        target_sess = sessions_list[curr_sess_idx]
        
        N_master = master_centroids.shape[0]
        N_target = target_sess['n_cells']
        
        # Build cost matrix
        cost_matrix = np.full((N_master, N_target), np.inf)
        
        # Compute shifts relative to the reference session
        # (Assuming the input centroids and footprints are ALREADY SHIFTED to reference coordinates!)
        target_tree = cKDTree(target_sess['centroids']) if N_target else None
        candidate_lists = (
            target_tree.query_ball_point(master_centroids, max_dist)
            if target_tree is not None else [[] for _ in range(N_master)]
        )
        for i, candidates in enumerate(candidate_lists):
            for j in candidates:
                d = np.linalg.norm(master_centroids[i] - target_sess['centroids'][j])
                overlap = compute_overlap(master_footprints[i], target_sess['sparse_footprints'][j], overlap_type)
                if overlap >= min_overlap:
                    cost_matrix[i, j] = cost_weight * (d / max_dist) + (1 - cost_weight) * (1 - overlap)
                        
        # Solve linear assignment
        pairs = solve_matching_pairs(cost_matrix, max_cost=1.0)
        
        # Map target cells to matching matrix
        new_col = np.full((N_master, 1), np.nan)
        matched_target = set()
        
        if len(pairs) > 0:
            for r, c in pairs:
                new_col[r, 0] = c
                matched_target.add(c)
                
        matching_matrix = np.hstack((matching_matrix, new_col))
        
        # Add unmatched cells to Master Union Map
        unmatched_targets = [c for c in range(N_target) if c not in matched_target]
        if unmatched_targets:
            # Append unmatched centroids and footprints to master list
            unmatched_centroids = target_sess['centroids'][unmatched_targets]
            unmatched_footprints = [target_sess['sparse_footprints'][c] for c in unmatched_targets]
            
            master_centroids = np.vstack((master_centroids, unmatched_centroids))
            master_footprints.extend(unmatched_footprints)
            
            # Add new rows to matching matrix (filled with NaNs, target cell ID in current session column)
            N_new = len(unmatched_targets)
            new_rows = np.full((N_new, s_order_idx + 1), np.nan)
            new_rows[:, s_order_idx] = unmatched_targets
            
            matching_matrix = np.vstack((matching_matrix, new_rows))
            
    # Sort matching matrix columns back to original chronological session list order
    revert_idx = np.argsort(ordered_idx)
    matching_matrix = matching_matrix[:, revert_idx]
    
    # Filter master cells by minimum activity (noise thresholding)
    active_counts = np.sum(~np.isnan(matching_matrix), axis=1)
    min_sess = 1
    keep_idx = active_counts >= min_sess
    
    matching_matrix = matching_matrix[keep_idx, :]
    master_centroids = master_centroids[keep_idx, :]
    master_footprints = [master_footprints[i] for i in range(len(master_footprints)) if keep_idx[i]]
    
    # Sort master cells by Y coordinate (top-to-bottom of the field of view)
    if master_centroids.shape[0] > 0:
        sort_idx = np.argsort(master_centroids[:, 1])
        matching_matrix = matching_matrix[sort_idx, :]
        master_centroids = master_centroids[sort_idx, :]
        master_footprints = [master_footprints[i] for i in sort_idx]
        
    return matching_matrix, master_centroids, master_footprints

def optimize_matching_parameters(sessions_list, target_count, overlap_type='cosine'):
    """Performs grid search to find parameters matching a target cell count."""
    max_dist_list = [8.0, 12.0, 16.0, 20.0, 24.0]
    min_overlap_list = [0.01, 0.03, 0.06, 0.10]
    cost_weight_list = [0.3, 0.5, 0.7]
    
    best_dist = 20.0
    best_overlap = 0.03
    best_weight = 0.5
    best_error = np.inf
    best_count = 0
    
    history = []
    
    for d in max_dist_list:
        for o in min_overlap_list:
            for w in cost_weight_list:
                mat, _, _ = match_cells_across_sessions(sessions_list, max_dist=d, min_overlap=o, cost_weight=w, overlap_type=overlap_type)
                n_cells = mat.shape[0]
                summary = summarize_matching(mat)
                err = abs(n_cells - target_count)
                
                history.append({
                    "max_dist": d,
                    "min_overlap": o,
                    "cost_weight": w,
                    "n_cells": n_cells,
                    "coverage_pct": summary["coverage_pct"],
                    "error": err
                })
                
                if err < best_error:
                    best_error = err
                    best_dist = d
                    best_overlap = o
                    best_weight = w
                    best_count = n_cells
                elif err == best_error:
                    if d < best_dist:
                        best_dist = d
                        best_overlap = o
                        best_weight = w
                        best_count = n_cells
                        
    return {
        "max_dist": best_dist,
        "min_overlap": best_overlap,
        "cost_weight": best_weight,
        "final_count": int(best_count),
        "history": history
    }

def compute_matching_distributions(matching_matrix, session_centroids, session_footprints, master_footprints, master_centroids, overlap_type='cosine'):
    """Computes distributions of within-cell/between-cell centroid distances and spatial overlaps."""
    within_distances = []
    between_distances = []
    within_overlaps = []
    between_overlaps = []
    
    M, S = matching_matrix.shape
    
    # 1. Within-cell distances (centroid drift) and overlaps across sessions
    for i in range(M):
        coords = []
        fps = []
        for s in range(S):
            local_idx = matching_matrix[i, s]
            if not np.isnan(local_idx) and local_idx >= 0:
                idx = int(local_idx)
                coords.append(session_centroids[s][idx])
                fps.append(session_footprints[s][idx])
                
        # Centroid drift (within-cell distance across sessions)
        if len(coords) >= 2:
            for k in range(len(coords)):
                for l in range(k + 1, len(coords)):
                    d = np.linalg.norm(coords[k] - coords[l])
                    within_distances.append(float(d))
                    
        # Within-cell footprint overlap across sessions
        if len(fps) >= 2:
            for k in range(len(fps)):
                for l in range(k + 1, len(fps)):
                    overlap = compute_overlap(fps[k], fps[l], overlap_type)
                    if overlap > 0.0:
                        within_overlaps.append(float(overlap))
                        
    # 2. Between-cell distances (different cells, up to 40px) and overlaps
    for i in range(M):
        for j in range(i + 1, M):
            d_cent = np.linalg.norm(master_centroids[i] - master_centroids[j])
            if d_cent <= 40.0:
                between_distances.append(float(d_cent))
                
            if d_cent <= 30.0:
                overlap = compute_overlap(master_footprints[i], master_footprints[j], overlap_type)
                if overlap > 0.0:
                    between_overlaps.append(float(overlap))
                    
    return within_distances, between_distances, within_overlaps, between_overlaps
