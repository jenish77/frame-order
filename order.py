#!/usr/bin/env python3
"""
Frame Order: Reconstructs the temporal sequence of video frames from a shuffled directory.
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist


def load_and_extract_features(filepath, target_size=(48, 48)):
    """
    Loads an image from filepath and extracts multi-scale perceptual features:
    1. Multi-scale spatial color downsamples (captures structural & color layout)
    2. Color histogram across RGB channels (captures scene color distribution)
    3. Gradient magnitude (Sobel differences) (captures edges and silhouettes)
    4. Mean luminance (captures lighting / fade transitions)
    """
    with Image.open(filepath) as raw_im:
        im = raw_im.convert('RGB')
        
    # Multi-scale thumbnails
    im_small = im.resize((24, 24), Image.Resampling.BILINEAR)
    im_med = im.resize(target_size, Image.Resampling.BILINEAR)
    
    arr_small = np.array(im_small, dtype=np.float32) / 255.0
    arr_med = np.array(im_med, dtype=np.float32) / 255.0
    
    # Color histograms (16 bins per channel)
    h_r, _ = np.histogram(arr_med[:, :, 0], bins=16, range=(0.0, 1.0))
    h_g, _ = np.histogram(arr_med[:, :, 1], bins=16, range=(0.0, 1.0))
    h_b, _ = np.histogram(arr_med[:, :, 2], bins=16, range=(0.0, 1.0))
    hist = np.concatenate([h_r, h_g, h_b]).astype(np.float32)
    hist /= (hist.sum() + 1e-7)
    
    # Gradients / Edge magnitude
    gray = 0.2989 * arr_med[:, :, 0] + 0.5870 * arr_med[:, :, 1] + 0.1140 * arr_med[:, :, 2]
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, :-1] = np.diff(gray, axis=1)
    gy[:-1, :] = np.diff(gray, axis=0)
    grad = np.sqrt(gx**2 + gy**2)
    norm_grad = np.linalg.norm(grad)
    if norm_grad > 1e-7:
        grad /= norm_grad
        
    return {
        'small': arr_small.flatten(),
        'med': arr_med.flatten(),
        'hist': hist,
        'grad': grad.flatten(),
        'mean_lum': float(np.mean(gray))
    }


def compute_distance_matrix(features):
    """
    Computes a symmetric distance matrix D between all pairs of frames
    combining multi-scale spatial pixel differences, color distribution, and gradients.
    """
    n = len(features)
    f_med = np.array([f['med'] for f in features])
    f_small = np.array([f['small'] for f in features])
    f_hist = np.array([f['hist'] for f in features])
    f_grad = np.array([f['grad'] for f in features])

    d_med = cdist(f_med, f_med, metric='cityblock')
    d_small = cdist(f_small, f_small, metric='cityblock')
    d_hist = cdist(f_hist, f_hist, metric='cityblock')
    d_grad = cdist(f_grad, f_grad, metric='cityblock')

    # Normalize components by their mean distance to balance scales
    mean_med = np.mean(d_med[d_med > 0]) if np.any(d_med > 0) else 1.0
    mean_small = np.mean(d_small[d_small > 0]) if np.any(d_small > 0) else 1.0
    mean_hist = np.mean(d_hist[d_hist > 0]) if np.any(d_hist > 0) else 1.0
    mean_grad = np.mean(d_grad[d_grad > 0]) if np.any(d_grad > 0) else 1.0

    d_med /= (mean_med + 1e-7)
    d_small /= (mean_small + 1e-7)
    d_hist /= (mean_hist + 1e-7)
    d_grad /= (mean_grad + 1e-7)

    # Weighted blend: Primary spatial match + multi-scale context + edge & color consistency
    D = 0.65 * d_med + 0.25 * d_small + 0.10 * d_hist + 0.10 * d_grad
    return D


def solve_tsp_held_karp(D):
    """
    Exact Hamiltonian Path via reduction to TSP with a dummy node.
    Used for small N (N <= 16).
    """
    n = len(D)
    n_aug = n + 1
    d_aug = np.zeros((n_aug, n_aug), dtype=np.float64)
    d_aug[1:, 1:] = D
    d_aug[0, :] = 0.0
    d_aug[:, 0] = 0.0

    memo = {}

    def tsp(mask, u):
        if mask == (1 << n_aug) - 1:
            return d_aug[u, 0], [0]
        state = (mask, u)
        if state in memo:
            return memo[state]
        best_cost = float('inf')
        best_path = None
        for v in range(n_aug):
            if not (mask & (1 << v)):
                c, p = tsp(mask | (1 << v), v)
                cost = d_aug[u, v] + c
                if cost < best_cost:
                    best_cost = cost
                    best_path = [v] + p
        memo[state] = (best_cost, best_path)
        return best_cost, best_path

    _, tour = tsp(1, 0)
    full_tour = [0] + tour
    # tour starts with 0 and ends with 0; intermediate nodes are 1..n
    path = [x - 1 for x in full_tour[1:-1]]
    return path


def solve_hamiltonian_path_heuristic(D, max_time=12.0):
    """
    High-performance Hamiltonian Path solver for larger N (N > 16).
    Employs:
    - Candidate lists (k-nearest neighbors)
    - Multi-start greedy path construction
    - 2-opt local search covering internal swaps, prefix reversals, and suffix reversals
    - Time-budgeted termination
    """
    n = len(D)
    k_cand = min(25, n - 1)
    
    # Precompute k-nearest neighbors
    k_nn = []
    for i in range(n):
        neighbors = sorted([(D[i, j], j) for j in range(n) if j != i])
        k_nn.append([j for _, j in neighbors[:k_cand]])

    best_path = None
    best_cost = float('inf')
    t_start = time.time()

    # Determine candidate start nodes:
    # Nodes with low minimum distance to neighbors and extremal nodes
    min_dists = [D[i, k_nn[i][0]] for i in range(n)]
    candidate_starts = sorted(range(n), key=lambda x: min_dists[x])[:min(20, n)]

    for start in candidate_starts:
        if time.time() - t_start > max_time and best_path is not None:
            break

        # Greedy construction
        path = [start]
        unvisited = set(range(n)) - {start}
        curr = start
        while unvisited:
            found = False
            for nxt in k_nn[curr]:
                if nxt in unvisited:
                    path.append(nxt)
                    unvisited.remove(nxt)
                    curr = nxt
                    found = True
                    break
            if not found:
                nxt = min(unvisited, key=lambda x: D[curr, x])
                path.append(nxt)
                unvisited.remove(nxt)
                curr = nxt

        # 2-opt refinement on open path
        improved = True
        step = 0
        while improved and (time.time() - t_start < max_time) and (step < 300):
            improved = False
            step += 1

            # 1. Internal segment reversal
            for i in range(1, n - 2):
                p_prev, p_i = path[i - 1], path[i]
                d_prev_i = D[p_prev, p_i]
                for j in range(i + 1, n - 1):
                    p_j, p_next = path[j], path[j + 1]
                    delta = (D[p_prev, p_j] + D[p_i, p_next]) - (d_prev_i + D[p_j, p_next])
                    if delta < -1e-7:
                        path[i:j + 1] = reversed(path[i:j + 1])
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # 2. Prefix reversal (changes start node)
            for j in range(1, n - 1):
                p_0, p_j, p_next = path[0], path[j], path[j + 1]
                delta = D[p_0, p_next] - D[p_j, p_next]
                if delta < -1e-7:
                    path[:j + 1] = reversed(path[:j + 1])
                    improved = True
                    break

            if improved:
                continue

            # 3. Suffix reversal (changes end node)
            for i in range(1, n - 1):
                p_prev, p_i, p_last = path[i - 1], path[i], path[-1]
                delta = D[p_prev, p_last] - D[p_prev, p_i]
                if delta < -1e-7:
                    path[i:] = reversed(path[i:])
                    improved = True
                    break

        cost = sum(D[path[k], path[k + 1]] for k in range(n - 1))
        if cost < best_cost:
            best_cost = cost
            best_path = list(path)

    return best_path


def orient_path(path, features):
    """
    Heuristic for temporal arrow of time:
    If one end has a clear intro fade (e.g. very dark/black start frames brightening into scene),
    ensure it starts at the intro.
    """
    if len(path) < 4:
        return path

    start_lum = np.mean([features[path[i]]['mean_lum'] for i in range(min(3, len(path)))])
    end_lum = np.mean([features[path[-1 - i]]['mean_lum'] for i in range(min(3, len(path)))])

    # If start is almost black (< 0.05) and end is bright scene, it's already an intro fade
    # If end is almost black and start is bright scene, it's an outro fade (reversed)
    if end_lum < 0.04 and start_lum > 0.15:
        return list(reversed(path))

    return path


def order_frames(frames_dir):
    """
    Reads all PNG frames from frames_dir and outputs their filenames in estimated temporal order.
    """
    if not os.path.exists(frames_dir):
        return []

    frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith('.png')]
    if not frame_files:
        return []

    # Single frame edge case
    if len(frame_files) == 1:
        return frame_files

    # Two frames edge case
    if len(frame_files) == 2:
        return frame_files

    # Load and extract features
    features = []
    valid_files = []
    for f in frame_files:
        full_path = os.path.join(frames_dir, f)
        try:
            feat = load_and_extract_features(full_path)
            features.append(feat)
            valid_files.append(f)
        except Exception as e:
            # If an image fails to load, keep filename for fallback
            pass

    n = len(valid_files)
    if n == 0:
        return frame_files
    if n == 1:
        return valid_files
    if n == 2:
        return valid_files

    # Compute pairwise distance matrix
    D = compute_distance_matrix(features)

    # Solve Hamiltonian Path
    if n <= 16:
        path = solve_tsp_held_karp(D)
    else:
        path = solve_hamiltonian_path_heuristic(D)

    # Orient path (arrow of time)
    path = orient_path(path, features)

    ordered_files = [valid_files[i] for i in path]

    # Safety check: Ensure every frame appears exactly once
    seen = set(ordered_files)
    for f in frame_files:
        if f not in seen:
            ordered_files.append(f)

    return ordered_files


def main():
    dir_path = sys.argv[1] if len(sys.argv) > 1 else '/frames'
    result = order_frames(dir_path)
    for fname in result:
        print(fname)


if __name__ == '__main__':
    main()
