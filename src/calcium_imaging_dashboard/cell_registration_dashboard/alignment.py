import cv2
import numpy as np
import SimpleITK as sitk
import scipy.optimize

from .alignment_models import AlignmentReference, AlignmentTransform, resolve_alignment_reference

def compose_warp_matrix_from_params(dx, dy, rotation, scale, cx=304.0, cy=304.0):
    """Compose the forward transform that moves a session into reference space."""
    T1 = np.eye(3, dtype=np.float32)
    T1[0, 2] = -cx
    T1[1, 2] = -cy
    
    R = np.eye(3, dtype=np.float32)
    rad = np.radians(rotation)
    R[0, 0] = scale * np.cos(rad)
    R[0, 1] = -scale * np.sin(rad)
    R[1, 0] = scale * np.sin(rad)
    R[1, 1] = scale * np.cos(rad)
    
    T2 = np.eye(3, dtype=np.float32)
    T2[0, 2] = cx + dx
    T2[1, 2] = cy + dy
    
    M_canvas = T2 @ R @ T1
    return M_canvas[0:2, 0:3]

def compute_centroids(spatial):
    """Calculates center of mass coordinates [x, y] for N x H x W footprints in Python.
    
    Returns a numpy array of shape (N, 2) with 1-based indexing matching MATLAB.
    """
    _, H, W = spatial.shape
    sums = np.sum(spatial, axis=(1, 2))
    x_weights = np.arange(1, W + 1, dtype=np.float64)
    y_weights = np.arange(1, H + 1, dtype=np.float64)
    x_moments = np.sum(np.sum(spatial, axis=1) * x_weights[None, :], axis=1)
    y_moments = np.sum(np.sum(spatial, axis=2) * y_weights[None, :], axis=1)
    centroids = np.zeros((spatial.shape[0], 2), dtype=np.float64)
    valid = sums > 0
    centroids[valid, 0] = x_moments[valid] / sums[valid]
    centroids[valid, 1] = y_moments[valid] / sums[valid]
    return centroids

def compute_ncc(A, B, downsample=True):
    """Computes Normalized Cross-Correlation matching the frontend's 64x64 downsampled Pearson Correlation."""
    A = np.nan_to_num(A)
    B = np.nan_to_num(B)
    
    if downsample and max(A.shape) > 64 and max(B.shape) > 64:
        A = cv2.resize(A, (64, 64), interpolation=cv2.INTER_AREA)
        B = cv2.resize(B, (64, 64), interpolation=cv2.INTER_AREA)
        
    A_zero = A - np.mean(A)
    B_zero = B - np.mean(B)
    norm_A = np.linalg.norm(A_zero)
    norm_B = np.linalg.norm(B_zero)
    den = norm_A * norm_B
    if den > 0:
        return float(np.sum(A_zero * B_zero) / den)
    return 0.0


def normalize_stack_image(image):
    """Robustly scale one overview image for multi-session coherence scoring."""
    values = np.nan_to_num(np.asarray(image, dtype=np.float64), copy=True)
    values[values < 0] = 0
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.5))
    if not high > low:
        high = float(np.max(values))
    if not high > low:
        return None
    values = np.clip((values - low) / (high - low), 0.0, 1.0)
    norm = float(np.linalg.norm(values))
    if norm <= 0:
        return None
    return values / norm


def compute_stack_coherence(images):
    """Return 0--1 constructive overlap coherence for an aligned image stack.

    Each session is robustly scaled and L2-normalized so its contribution is
    equal. Identical/co-localized stacks score 1; disjoint stacks score 0 after
    subtracting the finite-stack baseline (1 / number of sessions).
    """
    summed = None
    count = 0
    for image in images:
        normalized = normalize_stack_image(image)
        if normalized is None:
            continue
        if summed is None:
            summed = np.zeros_like(normalized, dtype=np.float64)
        summed += normalized
        count += 1
    if count < 2:
        return float("nan")
    raw = float(np.sum(summed * summed) / (count * count))
    baseline = 1.0 / count
    return float(np.clip((raw - baseline) / (1.0 - baseline), 0.0, 1.0))

def estimate_translation_phase(ref, target):
    """Estimates dx, dy translation shift between two images using Phase Correlation in FFT space."""
    ref = np.nan_to_num(ref).astype('float64')
    target = np.nan_to_num(target).astype('float64')
    
    # Subtract mean and normalize
    ref_zero = ref - np.mean(ref)
    if np.std(ref_zero) > 0:
        ref_zero /= np.std(ref_zero)
        
    target_zero = target - np.mean(target)
    if np.std(target_zero) > 0:
        target_zero /= np.std(target_zero)
        
    # Phase correlation
    (dx, dy), response = cv2.phaseCorrelate(ref_zero, target_zero)
    return dx, dy

def register_images(ref, target, mode='translation', demons_smoothing=1.5):
    """Registers target image to ref image using the specified alignment mode.
    
    Modes:
        - 'translation': translation only, returns 2x3 warp matrix and ncc
        - 'rigid': translation + rotation, returns 2x3 warp matrix and ncc
        - 'similarity': translation + rotation + scale, returns 2x3 warp matrix and ncc
        - 'non-rigid': deformable demons registration, returns (H, W, 2) displacement field and ncc
    """
    ref = np.nan_to_num(ref).astype('float32')
    target = np.nan_to_num(target).astype('float32')
    cx = target.shape[1] / 2.0
    cy = target.shape[0] / 2.0
    
    if mode == 'non-rigid':
        # Deformable registration via Demons (using SimpleITK)
        ref_sitk = sitk.GetImageFromArray(ref)
        target_sitk = sitk.GetImageFromArray(target)
        
        # Match histograms to align intensities
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        target_matched = matcher.Execute(target_sitk, ref_sitk)
        
        # Demons filter configuration
        demons = sitk.DemonsRegistrationFilter()
        demons.SetStandardDeviations(float(demons_smoothing))
        demons.SetNumberOfIterations(40)
        
        # Execute demons registration
        displacement_field = demons.Execute(ref_sitk, target_matched)
        
        # Convert displacement field back to NumPy (H, W, 2)
        displacement = sitk.GetArrayFromImage(displacement_field)
        
        # Warp target image using the displacement field to compute NCC
        warped = warp_image_non_rigid(target, displacement)
        ncc = compute_ncc(ref, warped)
        return displacement, ncc
    
    # Parametric models using dual-initialization SciPy Powell optimization
    # Run 1: Identity initialization [0.0, 0.0, ...]
    init_params_id = [0.0, 0.0] if mode == 'translation' else ([0.0, 0.0, 0.0] if mode == 'rigid' else [0.0, 0.0, 0.0, 1.0])
    def objective_id(params):
        if mode == 'translation':
            dx, dy = params[0], params[1]
            rotation = 0.0
            scale = 1.0
        elif mode == 'rigid':
            dx, dy = params[0], params[1]
            rotation = params[2]
            scale = 1.0
        else:
            dx, dy = params[0], params[1]
            rotation = params[2]
            scale = params[3]
        M = compose_warp_matrix_from_params(dx, dy, rotation, scale, cx=cx, cy=cy)
        warped = warp_image_rigid(target, M)
        return -compute_ncc(ref, warped)
        
    res_id = scipy.optimize.minimize(objective_id, init_params_id, method='Powell', options={'xtol': 1e-3, 'ftol': 1e-4})
    ncc_id = -res_id.fun
    
    # Run 2: Phase Correlation initialization [-dx_pc, -dy_pc, ...]
    dx_pc, dy_pc = estimate_translation_phase(ref, target)
    init_params_pc = [-dx_pc, -dy_pc] if mode == 'translation' else ([-dx_pc, -dy_pc, 0.0] if mode == 'rigid' else [-dx_pc, -dy_pc, 0.0, 1.0])
    def objective_pc(params):
        if mode == 'translation':
            dx, dy = params[0], params[1]
            rotation = 0.0
            scale = 1.0
        elif mode == 'rigid':
            dx, dy = params[0], params[1]
            rotation = params[2]
            scale = 1.0
        else:
            dx, dy = params[0], params[1]
            rotation = params[2]
            scale = params[3]
        M = compose_warp_matrix_from_params(dx, dy, rotation, scale, cx=cx, cy=cy)
        warped = warp_image_rigid(target, M)
        return -compute_ncc(ref, warped)
        
    res_pc = scipy.optimize.minimize(objective_pc, init_params_pc, method='Powell', options={'xtol': 1e-3, 'ftol': 1e-4})
    ncc_pc = -res_pc.fun
    
    # Select optimizer result with higher NCC
    if ncc_id >= ncc_pc:
        best_res = res_id
        best_ncc = ncc_id
    else:
        best_res = res_pc
        best_ncc = ncc_pc
        
    if mode == 'translation':
        dx, dy = best_res.x[0], best_res.x[1]
        rotation = 0.0
        scale = 1.0
    elif mode == 'rigid':
        dx, dy, rotation = best_res.x[0], best_res.x[1], best_res.x[2]
        scale = 1.0
    else:
        dx, dy, rotation, scale = best_res.x[0], best_res.x[1], best_res.x[2], best_res.x[3]
        
    warp_matrix = compose_warp_matrix_from_params(dx, dy, rotation, scale, cx=cx, cy=cy)
    return warp_matrix, best_ncc

def warp_image_rigid(img, warp_matrix):
    """Warps a 2D image using a 2x3 affine warp matrix."""
    img = np.nan_to_num(img).astype('float32')
    H, W = img.shape
    return cv2.warpAffine(img, warp_matrix, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

def warp_image_non_rigid(img, displacement):
    """Warps a 2D image using a non-rigid displacement field of shape (H, W, 2)."""
    img = np.nan_to_num(img).astype('float32')
    H, W = img.shape
    
    # Create coordinate grid
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    
    # SimpleITK displacement field arrays have channel order [dx, dy] at index 0 and 1
    map_x = (x_grid + displacement[:, :, 0]).astype('float32')
    map_y = (y_grid + displacement[:, :, 1]).astype('float32')
    
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def compute_alignment_nccs(
    mips,
    spatial_sums,
    active_index,
    reference_index,
    mode,
    active_transform,
    reference_transform=None,
    downsample=True,
):
    """Score both image sources in the same Direct or Sequential frame.

    For Sequential alignment, ``reference_index`` is the immediate crawl
    neighbour and both accumulated transforms place the pair in their common
    global frame before scoring.
    """
    if mode == "non-rigid":
        active_mip = warp_image_non_rigid(mips[active_index], active_transform)
        active_sf = warp_image_non_rigid(spatial_sums[active_index], active_transform)
        if reference_transform is None:
            reference_mip = mips[reference_index]
            reference_sf = spatial_sums[reference_index]
        else:
            reference_mip = warp_image_non_rigid(mips[reference_index], reference_transform)
            reference_sf = warp_image_non_rigid(spatial_sums[reference_index], reference_transform)
    else:
        active_mip = warp_image_rigid(mips[active_index], active_transform)
        active_sf = warp_image_rigid(spatial_sums[active_index], active_transform)
        if reference_transform is None:
            reference_mip = mips[reference_index]
            reference_sf = spatial_sums[reference_index]
        else:
            reference_mip = warp_image_rigid(mips[reference_index], reference_transform)
            reference_sf = warp_image_rigid(spatial_sums[reference_index], reference_transform)
    return {
        "mip_ncc": float(compute_ncc(reference_mip, active_mip, downsample)),
        "footprints_ncc": float(compute_ncc(reference_sf, active_sf, downsample)),
    }


def compose_displacement_fields(accumulated, adjacent):
    """Compose ref->neighbor and neighbor->target remap displacement fields.

    The dashboard's displacement arrays are used as OpenCV backward maps:
    output(x) samples input(x + displacement(x)). For a crawl, the adjacent
    field must therefore be sampled at the coordinates produced by the already
    accumulated field before the vectors are added.
    """
    if accumulated is None:
        return np.array(adjacent, dtype=np.float32, copy=True)
    H, W, _ = accumulated.shape
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (x_grid + accumulated[:, :, 0]).astype(np.float32)
    map_y = (y_grid + accumulated[:, :, 1]).astype(np.float32)
    sampled_x = cv2.remap(
        adjacent[:, :, 0].astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    sampled_y = cv2.remap(
        adjacent[:, :, 1].astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    result = np.array(accumulated, dtype=np.float32, copy=True)
    result[:, :, 0] += sampled_x
    result[:, :, 1] += sampled_y
    return result

def warp_footprints_rigid(footprints, warp_matrix):
    """Warps a stack of spatial footprints of shape (N, H, W) using a 2x3 warp matrix."""
    N, H, W = footprints.shape
    warped = np.zeros_like(footprints)
    for i in range(N):
        warped[i] = warp_image_rigid(footprints[i], warp_matrix)
    return warped

def warp_footprints_non_rigid(footprints, displacement):
    """Warps a stack of spatial footprints of shape (N, H, W) using a displacement field."""
    N, H, W = footprints.shape
    warped = np.zeros_like(footprints)
    for i in range(N):
        warped[i] = warp_image_non_rigid(footprints[i], displacement)
    return warped
