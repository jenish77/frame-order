# Notes on Frame Order Solution

## 1. Approach and Formulation

The problem of reconstructing the original temporal order of shuffled video frames is formulated as finding a **Minimum Weight Hamiltonian Path** on a complete graph $G = (V, E)$, where each vertex $v \in V$ represents a video frame, and edge weights $D(i, j)$ quantify the perceptual dissimilarity between frame $i$ and frame $j$.

Because video frames captured at regular frame rates (24–60 fps) exhibit high temporal coherence, consecutive frames $(f_t, f_{t+1})$ minimize visual dissimilarity relative to non-consecutive pairs.

### Mathematical Reduction
To solve the open Hamiltonian path problem:
- For small videos ($N \le 16$): We reduce the open path problem to the Traveling Salesperson Problem (TSP) by introducing a virtual "dummy" node $0$ with edge weights $D(0, i) = D(i, 0) = 0$ for all frames $i \in \{1, \dots, N\}$. Any TSP tour visiting $\{0, \pi_1, \pi_2, \dots, \pi_N, 0\}$ has total cost:
  $$C = 0 + \sum_{k=1}^{N-1} D(\pi_k, \pi_{k+1}) + 0 = \sum_{k=1}^{N-1} D(\pi_k, \pi_{k+1})$$
  which corresponds exactly to the open Hamiltonian path cost. The dummy node automatically selects the optimal start and end frames. We solve this to exact optimality using the Held-Karp dynamic programming algorithm in $O(N^2 2^N)$ time.
- For larger videos ($N > 16$): We use an open-path multi-start heuristic combining candidate-list greedy construction with iterative 2-opt local search. In addition to internal segment reversals, the search explicitly tests prefix reversals (which change the start frame) and suffix reversals (which change the end frame), bounding runtime within a strict time budget.

---

## 2. Feature Extraction & Perceptual Distance Metric

Relying solely on full-resolution pixel differences can be sensitive to small pixel noise and computational bottlenecks. Instead, we compute a balanced multi-modal representation:
1. **Multi-Scale Spatial Downsampling (24×24 and 48×48)**:
   - Captures both global scene layout and localized structural features.
   - Resizing provides spatial smoothing against sub-pixel jitter and sensor noise.
2. **Color Distribution Histograms**:
   - 16-bin per-channel RGB histograms capture global color palettes, remaining consistent even when objects move rapidly across the frame.
3. **Sobel Gradient / Edge Magnitude**:
   - Approximates structural silhouettes and contours, providing robustness against minor illumination shifts.
4. **L1 (Cityblock) Distance with Mean Normalization**:
   - Normalized cityblock distance is robust against outlier pixel shifts and avoids squaring large isolated differences.

The final composite distance matrix is:
$$D = 0.65 \cdot D_{\text{med}} + 0.25 \cdot D_{\text{small}} + 0.10 \cdot D_{\text{hist}} + 0.10 \cdot D_{\text{grad}}$$

---

## 3. Directionality (Arrow of Time)

Because $D(i, j)$ is symmetric, the reverse sequence $\pi^{rev} = (\pi_N, \pi_{N-1}, \dots, \pi_1)$ has the same total path cost as $\pi = (\pi_1, \dots, \pi_N)$. To determine the forward orientation:
- We analyze boundary luminance transitions. Videos beginning with an intro fade (opening with near-black frames that transition into scene lighting) are oriented to ensure the fade begins at $t=0$.

---

## 4. Where It Still Goes Wrong (Known Limitations)

1. **Time-Reversible and Symmetrical Motion**:
   - In scenes without asymmetric physical cues (e.g. constant camera panning across a landscape, a bouncing pendulum, or zooming in vs. zooming out without text), the forward and backward paths have identical path costs. Without semantic understanding or text/audio, the direction of time remains mathematically ambiguous.
2. **Videos with Hard Scene Cuts**:
   - If a video contains multiple disparate shots (e.g. shot A then shot B), intra-shot distances are very small while the cut between shot A and shot B incurs a high distance. Although the TSP naturally clusters frames of shot A and shot B separately, determining which entire shot occurred first requires higher-level narrative understanding.
3. **Periodic and Cyclical Motion**:
   - In videos containing repetitive cycles (e.g., a rotating wheel or walking loop where frame $t$ looks identical to frame $t+30$), greedy heuristics can get trapped in shortcutting cycles unless the full path global cost is evaluated.
4. **Static / Paused Video Frames**:
   - Duplicate or near-identical frames form zero-distance cliques. While all frames are correctly preserved in the output, their internal relative order during the pause is arbitrary.
