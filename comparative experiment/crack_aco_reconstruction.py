import os
import math
import time
import random
import warnings
import logging
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize, thin
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    # --- Preprocessing ---
    "binarize_threshold": 128,       # Binarisation threshold
    "do_skeletonize": True,          # Whether to thin the input skeleton again

    # --- Endpoint detection ---
    "direction_window": 10,          # Number of pixels traced inward from an endpoint for local direction estimation

    # --- Candidate connection pair generation ---
    "max_gap": 50,                   # Maximum allowed connection distance (pixels)
    "min_gap": 3,                    # Minimum connection distance (too close to connect, possibly already adjacent)
    "max_angle_deg": 60,             # Candidate pairs with a direction angle greater than this value are excluded directly (degrees)
    "interference_buffer": 3,        # Buffer width for interference penalty detection (pixels)
    "max_candidates_per_endpoint": 8, # Maximum number of candidate pairs retained for each endpoint

    # --- Scoring weights ---
    "w_dist": 0.25,                  # Weight of the distance score
    "w_dir": 0.35,                   # Weight of directional consistency
    "w_smooth": 0.20,                # Weight of smoothness/continuity
    "w_clear": 0.10,                 # Weight of the interference penalty
    "w_topo": 0.10,                  # Weight of topological rationality

    # --- Scoring parameters ---
    "sigma_d": 20.0,                 # Decay coefficient for the distance score
    "sigma_theta": 30.0,             # Decay coefficient for the direction angle score (degrees)

    # --- ACO parameters ---
    "ant_count": 30,                 # Number of ants
    "max_iter": 50,                  # Maximum number of iterations
    "alpha": 1.0,                    # Importance coefficient of pheromone
    "beta": 2.0,                     # Importance coefficient of heuristic information
    "rho": 0.1,                      # Pheromone evaporation rate
    "tau_init": 1.0,                 # Initial pheromone value
    "tau_min": 0.01,                 # Minimum pheromone value (prevents premature convergence)
    "tau_max": 10.0,                 # Maximum pheromone value
    "q0": 0.6,                       # Greedy selection probability (ACS variant)

    # --- Connection line generation ---
    "use_smooth_curve": False,       # Whether to use a smooth curve (False = Bresenham straight line)
}

def load_and_preprocess_image(
    image_path: str,
    binarize_threshold: int = 128,
    do_skeletonize: bool = True
) -> np.ndarray:

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到输入文件：{image_path}")

    logger.info(f"读取图像：{image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")

    logger.info(f"图像尺寸：{img.shape}，像素值范围：[{img.min()}, {img.max()}]")

    # Binarisation
    _, binary = cv2.threshold(img, binarize_threshold, 255, cv2.THRESH_BINARY)

    # Optional: perform skeletonisation again to ensure single-pixel width
    if do_skeletonize:
        logger.info("执行骨架细化（skeletonize）...")
        bool_img = binary > 0
        skel = skeletonize(bool_img)
        skeleton = (skel * 255).astype(np.uint8)
    else:
        skeleton = binary.copy()

    white_pixels = np.sum(skeleton > 0)
    logger.info(f"预处理完成，骨架白色像素数：{white_pixels}")
    return skeleton

def extract_connected_components(skeleton: np.ndarray) -> Tuple[np.ndarray, int]:
    binary = (skeleton > 0).astype(np.uint8)
    num_labels, labeled = cv2.connectedComponents(binary, connectivity=8)
    logger.info(f"连通域分析完成，共 {num_labels - 1} 个裂隙片段")
    return labeled, num_labels - 1  # Subtract the background label

def detect_endpoints_and_branchpoints(
    skeleton: np.ndarray
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:

    binary = (skeleton > 0).astype(np.uint8)
    h, w = binary.shape

    # Use a convolution kernel to quickly count white pixels in the 8-neighbourhood
    kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(binary, -1, kernel.astype(np.float32))
    neighbor_count = neighbor_count.astype(np.int32)

    # Count only at skeleton pixels
    skeleton_mask = binary > 0
    endpoint_mask = skeleton_mask & (neighbor_count == 1)
    branch_mask = skeleton_mask & (neighbor_count >= 3)

    endpoints = list(zip(*np.where(endpoint_mask)))
    branchpoints = list(zip(*np.where(branch_mask)))

    logger.info(f"检测到断点（端点）：{len(endpoints)} 个，分叉点：{len(branchpoints)} 个")
    return endpoints, branchpoints

def _trace_skeleton_path(
    skeleton: np.ndarray,
    start: Tuple[int, int],
    max_steps: int = 15
) -> List[Tuple[int, int]]:
    binary = (skeleton > 0)
    h, w = binary.shape
    path = [start]
    visited = {start}
    current = start

    for _ in range(max_steps):
        r, c = current
        # Traverse the 8-neighbourhood
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if binary[nr, nc] and (nr, nc) not in visited:
                        neighbors.append((nr, nc))

        if not neighbors:
            break

        # Select the first unvisited neighbour (simple tracing strategy)
        next_pt = neighbors[0]
        path.append(next_pt)
        visited.add(next_pt)
        current = next_pt

    return path


def estimate_endpoint_direction(
    skeleton: np.ndarray,
    endpoint: Tuple[int, int],
    direction_window: int = 10
) -> np.ndarray:
    path = _trace_skeleton_path(skeleton, endpoint, max_steps=direction_window)

    if len(path) < 2:
        # Return the default direction if tracing fails
        return np.array([1.0, 0.0])

    # Convert to a coordinate array (N, 2) in (row, col) format
    pts = np.array(path, dtype=np.float64)

    if len(pts) < 2:
        return np.array([1.0, 0.0])

    # PCA: use the principal component direction
    pts_centered = pts - pts.mean(axis=0)
    if len(pts_centered) >= 2:
        cov = np.cov(pts_centered.T)
        if cov.ndim == 0:
            cov = np.array([[cov, 0], [0, 0]])
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Use the eigenvector corresponding to the largest eigenvalue
        main_dir = eigvecs[:, np.argmax(eigvals)]
    else:
        # If there are only two points, calculate the direction directly
        main_dir = pts[1] - pts[0]

    # Normalisation
    norm = np.linalg.norm(main_dir)
    if norm < 1e-8:
        return np.array([1.0, 0.0])
    main_dir = main_dir / norm

    # Adjust the direction so that the vector points from the endpoint to the "interior" (i.e., the tracing direction),
    # then reverse it to obtain the direction "extending outward from the endpoint"
    if len(path) >= 2:
        inner_dir = np.array(path[-1], dtype=np.float64) - np.array(path[0], dtype=np.float64)
        if np.dot(main_dir, inner_dir) < 0:
            main_dir = -main_dir
        # Final direction = the opposite of the inward direction (outward)
        main_dir = -main_dir

    norm = np.linalg.norm(main_dir)
    if norm < 1e-8:
        return np.array([1.0, 0.0])
    return main_dir / norm

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_val = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


def line_interference_penalty(
    skeleton: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    buffer_width: int = 3,
    component_label1: int = -1,
    component_label2: int = -1,
    labeled_map: Optional[np.ndarray] = None
) -> float:
    h, w = skeleton.shape

    # Generate pixels on the connection line (Bresenham)
    line_pixels = _bresenham_line(p1, p2)
    if not line_pixels:
        return 0.0

    # Create the buffer mask
    line_mask = np.zeros((h, w), dtype=np.uint8)
    for (r, c) in line_pixels:
        if 0 <= r < h and 0 <= c < w:
            line_mask[r, c] = 1

    # Dilate to obtain the buffer area
    kernel_size = 2 * buffer_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    buffer_mask = cv2.dilate(line_mask, kernel)

    # Count white skeleton pixels within the buffer area
    skel_binary = (skeleton > 0).astype(np.uint8)
    interfering = buffer_mask & skel_binary

    if labeled_map is not None and component_label1 > 0 and component_label2 > 0:
        # Exclude pixels belonging to the connected components of the two endpoints themselves
        own_mask = (labeled_map == component_label1) | (labeled_map == component_label2)
        interfering = interfering & (~own_mask.astype(bool))

    interference_count = np.sum(interfering)
    # Normalise by the connection line length
    line_length = max(len(line_pixels), 1)
    penalty = min(interference_count / (line_length * buffer_width * 2 + 1e-8), 1.0)
    return float(penalty)


def _bresenham_line(
    p1: Tuple[int, int],
    p2: Tuple[int, int]
) -> List[Tuple[int, int]]:
    r1, c1 = p1
    r2, c2 = p2
    pixels = []
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r2 > r1 else -1
    sc = 1 if c2 > c1 else -1
    err = dr - dc

    r, c = r1, c1
    while True:
        pixels.append((r, c))
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return pixels


def evaluate_connection_pair(
    skeleton: np.ndarray,
    ep1: Tuple[int, int],
    ep2: Tuple[int, int],
    dir1: np.ndarray,
    dir2: np.ndarray,
    label1: int,
    label2: int,
    labeled_map: np.ndarray,
    params: dict
) -> float:
    r1, c1 = ep1
    r2, c2 = ep2

    # Connection vector (from ep1 to ep2)
    conn_vec = np.array([r2 - r1, c2 - c1], dtype=np.float64)
    dist = np.linalg.norm(conn_vec)
    if dist < 1e-8:
        return 0.0

    conn_dir = conn_vec / dist  # Normalise the connection direction

    sigma_d = params.get("sigma_d", 20.0)
    S_dist = math.exp(-dist / sigma_d)

    sigma_theta = params.get("sigma_theta", 30.0)
    theta1 = angle_between_vectors(dir1, conn_dir)
    theta2 = angle_between_vectors(dir2, -conn_dir)

    theta1 = min(theta1, 180.0 - theta1)
    theta2 = min(theta2, 180.0 - theta2)
    S_dir = math.exp(-(theta1 + theta2) / (2 * sigma_theta))

    max_angle = params.get("max_angle_deg", 60.0)
    avg_angle = (theta1 + theta2) / 2.0

    S_smooth = max(0.0, 1.0 - avg_angle / max_angle)

    buffer_width = params.get("interference_buffer", 3)
    interference_penalty = line_interference_penalty(
        skeleton, ep1, ep2, buffer_width, label1, label2, labeled_map
    )
    S_clear = 1.0 - interference_penalty  # Less interference gives a higher score

    if label1 != label2:
        S_topo = 1.0
    else:

        S_topo = max(0.2, 1.0 - math.exp(-dist / (sigma_d * 2)))

    w_dist = params.get("w_dist", 0.25)
    w_dir = params.get("w_dir", 0.35)
    w_smooth = params.get("w_smooth", 0.20)
    w_clear = params.get("w_clear", 0.10)
    w_topo = params.get("w_topo", 0.10)

    score = (w_dist * S_dist +
             w_dir * S_dir +
             w_smooth * S_smooth +
             w_clear * S_clear +
             w_topo * S_topo)

    return float(np.clip(score, 0.0, 1.0))


def generate_candidate_pairs(
    skeleton: np.ndarray,
    endpoints: List[Tuple[int, int]],
    directions: Dict[Tuple[int, int], np.ndarray],
    endpoint_labels: Dict[Tuple[int, int], int],
    labeled_map: np.ndarray,
    params: dict
) -> List[Dict]:

    max_gap = params.get("max_gap", 50)
    min_gap = params.get("min_gap", 3)
    max_angle_deg = params.get("max_angle_deg", 60)
    max_per_ep = params.get("max_candidates_per_endpoint", 8)

    n = len(endpoints)
    logger.info(f"开始生成候选连接对，断点数：{n}...")

    # Accelerate using spatial indexing (simple KD-tree approach)
    ep_array = np.array(endpoints, dtype=np.float64)

    candidates = []
    # Record the number of candidate pairs for each endpoint (to limit the upper bound per endpoint)
    ep_candidate_count = [0] * n

    for i in range(n):
        if ep_candidate_count[i] >= max_per_ep:
            continue

        ep1 = endpoints[i]
        dir1 = directions[ep1]
        label1 = endpoint_labels[ep1]

        # Calculate distances to all other endpoints for rapid screening
        dists = np.linalg.norm(ep_array - ep_array[i], axis=1)
        sorted_indices = np.argsort(dists)

        count_added = 0
        for j_idx in sorted_indices:
            j = int(j_idx)
            if j <= i:
                continue
            if ep_candidate_count[j] >= max_per_ep:
                continue

            d = float(dists[j])
            if d < min_gap:
                continue
            if d > max_gap:
                break  # Already sorted, so the following ones are farther away

            ep2 = endpoints[j]
            dir2 = directions[ep2]
            label2 = endpoint_labels[ep2]

            # Rapid direction pre-screening
            conn_vec = np.array([ep2[0] - ep1[0], ep2[1] - ep1[1]], dtype=np.float64)
            conn_dir = conn_vec / (np.linalg.norm(conn_vec) + 1e-8)
            theta1 = angle_between_vectors(dir1, conn_dir)
            theta2 = angle_between_vectors(dir2, -conn_dir)
            theta1 = min(theta1, 180.0 - theta1)
            theta2 = min(theta2, 180.0 - theta2)

            if theta1 > max_angle_deg or theta2 > max_angle_deg:
                continue

            # Calculate the comprehensive score
            score = evaluate_connection_pair(
                skeleton, ep1, ep2, dir1, dir2,
                label1, label2, labeled_map, params
            )

            if score > 0.01:  # Filter out candidate pairs with extremely low scores
                candidates.append({
                    'ep1': ep1,
                    'ep2': ep2,
                    'dist': d,
                    'score': score,
                    'label1': label1,
                    'label2': label2,
                    'idx1': i,
                    'idx2': j
                })
                ep_candidate_count[i] += 1
                ep_candidate_count[j] += 1
                count_added += 1

            if count_added >= max_per_ep:
                break

    # Sort in descending order by score
    candidates.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"候选连接对生成完成，共 {len(candidates)} 对")
    return candidates

class CrackSkeletonACOReconstructor:

    def __init__(self, candidates: List[Dict], params: dict):

        self.candidates = candidates
        self.params = params
        self.n_edges = len(candidates)

        if self.n_edges == 0:
            logger.warning("候选连接对为空，ACO无需运行")
            self.best_solution = []
            self.best_score = 0.0
            return

        # ACO parameters
        self.ant_count = params.get("ant_count", 30)
        self.max_iter = params.get("max_iter", 50)
        self.alpha = params.get("alpha", 1.0)
        self.beta = params.get("beta", 2.0)
        self.rho = params.get("rho", 0.1)
        self.tau_init = params.get("tau_init", 1.0)
        self.tau_min = params.get("tau_min", 0.01)
        self.tau_max = params.get("tau_max", 10.0)
        self.q0 = params.get("q0", 0.6)

        # Initialise pheromones
        self.pheromones = np.full(self.n_edges, self.tau_init, dtype=np.float64)

        # Heuristic information (score of each candidate edge)
        self.heuristics = np.array([c['score'] for c in candidates], dtype=np.float64)
        # Prevent division by zero
        self.heuristics = np.clip(self.heuristics, 1e-8, None)

        # Build an endpoint-to-candidate-edge index (to accelerate conflict detection)
        self._build_endpoint_index()

        self.best_solution: List[int] = []  # Best solution: a list of candidate edge indices
        self.best_score: float = 0.0
        self.score_history: List[float] = []

    def _build_endpoint_index(self):

        self.endpoint_to_edges: Dict[Tuple[int, int], List[int]] = {}
        for edge_idx, cand in enumerate(self.candidates):
            ep1 = cand['ep1']
            ep2 = cand['ep2']
            self.endpoint_to_edges.setdefault(ep1, []).append(edge_idx)
            self.endpoint_to_edges.setdefault(ep2, []).append(edge_idx)

    def _check_crossing(self, edge_idx1: int, edge_idx2: int) -> bool:

        c1 = self.candidates[edge_idx1]
        c2 = self.candidates[edge_idx2]
        return _segments_intersect(c1['ep1'], c1['ep2'], c2['ep1'], c2['ep2'])

    def construct_solution_by_ant(self) -> List[int]:

        if self.n_edges == 0:
            return []

        solution: List[int] = []
        used_endpoints: Set[Tuple[int, int]] = set()

        # Calculate the selection probability of each edge (proportional to pheromone^alpha * heuristic^beta)
        tau_alpha = np.power(np.clip(self.pheromones, 1e-8, None), self.alpha)
        eta_beta = np.power(self.heuristics, self.beta)
        weights = tau_alpha * eta_beta

        # Randomly select the starting edge (probability-weighted)
        total_weight = weights.sum()
        if total_weight < 1e-8:
            return []

        # Generate a random visiting order (sorted by weight with slight random perturbation)
        noise = np.random.uniform(0.8, 1.2, self.n_edges)
        shuffled_weights = weights * noise
        order = np.argsort(-shuffled_weights)  # Sort in descending order

        for edge_idx in order:
            edge_idx = int(edge_idx)
            cand = self.candidates[edge_idx]
            ep1 = cand['ep1']
            ep2 = cand['ep2']

            # Constraint 1: whether the endpoint has already been used
            if ep1 in used_endpoints or ep2 in used_endpoints:
                continue

            # Constraint 2: whether it intersects with selected edges
            has_crossing = False
            for selected_idx in solution:
                if self._check_crossing(edge_idx, selected_idx):
                    has_crossing = True
                    break
            if has_crossing:
                continue

            # ACS greedy strategy: directly select the current best edge with probability q0
            q = random.random()
            w = weights[edge_idx]
            if q < self.q0:
                # Greedy selection: accept directly
                accept = True
            else:
                # Roulette selection: decide acceptance based on weight (acceptance probability is proportional to weight)
                accept_prob = min(w / (total_weight * 0.1 + 1e-8), 1.0)
                accept = random.random() < accept_prob

            if accept:
                solution.append(edge_idx)
                used_endpoints.add(ep1)
                used_endpoints.add(ep2)

                # Local pheromone update (ACS strategy: reduce pheromone on selected edges to encourage exploration by other ants)
                self.pheromones[edge_idx] = max(
                    self.tau_min,
                    (1 - self.rho) * self.pheromones[edge_idx] + self.rho * self.tau_init
                )

        return solution

    def evaluate_solution(self, solution: List[int]) -> float:

        if not solution:
            return 0.0

        total_score = 0.0
        for edge_idx in solution:
            cand = self.candidates[edge_idx]
            total_score += cand['score']

        # Crossing penalty: detect whether intersecting connections exist within the solution
        crossing_penalty = 0.0
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                if self._check_crossing(solution[i], solution[j]):
                    crossing_penalty += 0.5  # Deduct 0.5 points for each crossing pair

        return total_score - crossing_penalty

    def update_pheromones(self, all_solutions: List[List[int]], all_scores: List[float]):

        # Evaporation
        self.pheromones = (1 - self.rho) * self.pheromones
        self.pheromones = np.clip(self.pheromones, self.tau_min, self.tau_max)

        if not all_solutions:
            return

        # Find the best solution in this iteration
        best_idx = int(np.argmax(all_scores))
        best_sol = all_solutions[best_idx]
        best_score = all_scores[best_idx]

        if best_score <= 0:
            return

        # Increase pheromone on edges in the best solution
        delta = best_score / max(len(best_sol), 1)
        for edge_idx in best_sol:
            self.pheromones[edge_idx] = min(
                self.tau_max,
                self.pheromones[edge_idx] + delta
            )

        # If the global best solution is updated, also reinforce pheromone on its edges
        if best_score > self.best_score:
            self.best_score = best_score
            self.best_solution = best_sol[:]
            extra_delta = best_score / max(len(best_sol), 1)
            for edge_idx in best_sol:
                self.pheromones[edge_idx] = min(
                    self.tau_max,
                    self.pheromones[edge_idx] + extra_delta
                )

    def run(self) -> Tuple[List[Dict], float]:

        if self.n_edges == 0:
            logger.info("没有候选连接对，跳过ACO")
            return [], 0.0

        logger.info(f"开始ACO优化：{self.ant_count}只蚂蚁，{self.max_iter}轮迭代")
        start_time = time.time()

        for iteration in range(self.max_iter):
            all_solutions = []
            all_scores = []

            for ant in range(self.ant_count):
                solution = self.construct_solution_by_ant()
                score = self.evaluate_solution(solution)
                all_solutions.append(solution)
                all_scores.append(score)

            self.update_pheromones(all_solutions, all_scores)
            self.score_history.append(self.best_score)

            if (iteration + 1) % 10 == 0 or iteration == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"  迭代 {iteration + 1}/{self.max_iter}，"
                    f"当前最优分：{self.best_score:.4f}，"
                    f"选中连接数：{len(self.best_solution)}，"
                    f"耗时：{elapsed:.1f}s"
                )

        total_time = time.time() - start_time
        logger.info(f"ACO完成，总耗时：{total_time:.1f}s")
        logger.info(f"最终最优分：{self.best_score:.4f}，选中连接数：{len(self.best_solution)}")

        best_connections = [self.candidates[idx] for idx in self.best_solution]
        return best_connections, self.best_score

def _cross_product_2d(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p, q, r) -> bool:
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
        return True
    return False


def _segments_intersect(
    p1: Tuple[int, int], p2: Tuple[int, int],
    p3: Tuple[int, int], p4: Tuple[int, int]
) -> bool:

    # Shared endpoints are not considered intersections
    if p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4:
        return False

    d1 = _cross_product_2d(p3, p4, p1)
    d2 = _cross_product_2d(p3, p4, p2)
    d3 = _cross_product_2d(p1, p2, p3)
    d4 = _cross_product_2d(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases
    if d1 == 0 and _on_segment(p3, p1, p4):
        return True
    if d2 == 0 and _on_segment(p3, p2, p4):
        return True
    if d3 == 0 and _on_segment(p1, p3, p2):
        return True
    if d4 == 0 and _on_segment(p1, p4, p2):
        return True

    return False

def draw_connection(
    skeleton: np.ndarray,
    ep1: Tuple[int, int],
    ep2: Tuple[int, int],
    direction1: Optional[np.ndarray] = None,
    direction2: Optional[np.ndarray] = None,
    use_smooth_curve: bool = False
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    new_skeleton = skeleton.copy()
    h, w = skeleton.shape

    if use_smooth_curve and direction1 is not None and direction2 is not None:
        new_pixels = _draw_smooth_curve(ep1, ep2, direction1, direction2, h, w)
    else:
        new_pixels = _bresenham_line(ep1, ep2)

    for (r, c) in new_pixels:
        if 0 <= r < h and 0 <= c < w:
            new_skeleton[r, c] = 255

    return new_skeleton, new_pixels


def _draw_smooth_curve(
    ep1: Tuple[int, int],
    ep2: Tuple[int, int],
    dir1: np.ndarray,
    dir2: np.ndarray,
    h: int, w: int,
    n_points: int = 50
) -> List[Tuple[int, int]]:

    dist = np.linalg.norm(np.array(ep2) - np.array(ep1))
    ctrl_scale = dist * 0.4  # Offset distance of control points

    # Control points: offset outward from the endpoints along the directions
    # Note: dir is in (row, col) format
    p0 = np.array([ep1[0], ep1[1]], dtype=np.float64)
    p3 = np.array([ep2[0], ep2[1]], dtype=np.float64)

    # dir1 points outward, and the control point lies in the extension direction of ep1
    p1 = p0 + dir1 * ctrl_scale
    # dir2 points outward, which is the reverse direction from the perspective of ep2
    p2 = p3 + dir2 * ctrl_scale

    # Cubic Bezier curve sampling
    pixels = set()
    for t in np.linspace(0, 1, n_points):
        # Bezier formula
        pt = ((1-t)**3 * p0 +
              3*(1-t)**2*t * p1 +
              3*(1-t)*t**2 * p2 +
              t**3 * p3)
        r, c = int(round(pt[0])), int(round(pt[1]))
        if 0 <= r < h and 0 <= c < w:
            pixels.add((r, c))

    return list(pixels)


def reconstruct_skeleton(
    skeleton: np.ndarray,
    best_connections: List[Dict],
    directions: Dict[Tuple[int, int], np.ndarray],
    use_smooth_curve: bool = False
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:

    new_skeleton = skeleton.copy()
    all_new_pixels = []

    for conn in best_connections:
        ep1 = conn['ep1']
        ep2 = conn['ep2']
        dir1 = directions.get(ep1, np.array([1.0, 0.0]))
        dir2 = directions.get(ep2, np.array([1.0, 0.0]))

        new_skeleton, new_pixels = draw_connection(
            new_skeleton, ep1, ep2, dir1, dir2, use_smooth_curve
        )
        all_new_pixels.append(new_pixels)
        logger.info(
            f"  连接：{ep1} -> {ep2}，"
            f"距离：{conn['dist']:.1f}px，"
            f"评分：{conn['score']:.3f}，"
            f"新增像素：{len(new_pixels)}"
        )

    return new_skeleton, all_new_pixels

def visualize_results(
    original_skeleton: np.ndarray,
    reconstructed_skeleton: np.ndarray,
    endpoints: List[Tuple[int, int]],
    best_connections: List[Dict],
    all_new_pixels: List[List[Tuple[int, int]]],
    output_dir: str,
    input_filename: str,
    aco_score_history: Optional[List[float]] = None
):

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_filename))[0]

    # --- Output 1: reconstructed binary skeleton image ---
    out_skel_path = os.path.join(output_dir, f"{base_name}_reconstructed.png")
    cv2.imwrite(out_skel_path, reconstructed_skeleton)
    logger.info(f"重构骨架图已保存：{out_skel_path}")

    # --- Output 2: colour visualisation image ---
    h, w = original_skeleton.shape

    # Create an RGB colour image (black background, white original skeleton)
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Original skeleton: white
    mask_orig = original_skeleton > 0
    color_img[mask_orig] = [200, 200, 200]  # Grey-white indicates the original skeleton

    # Newly added connection lines: green
    for new_pixels in all_new_pixels:
        for (r, c) in new_pixels:
            if 0 <= r < h and 0 <= c < w:
                color_img[r, c] = [0, 255, 0]  # Green

    # Endpoints: red (slightly larger, marked with 5x5 blocks)
    ep_set = set([conn['ep1'] for conn in best_connections] +
                 [conn['ep2'] for conn in best_connections])
    all_ep_set = set(map(tuple, endpoints))

    for ep in all_ep_set:
        r, c = ep
        # Unconnected endpoints: red
        r0 = max(0, r - 2)
        r1 = min(h, r + 3)
        c0 = max(0, c - 2)
        c1 = min(w, c + 3)
        if ep in ep_set:
            color_img[r0:r1, c0:c1] = [0, 0, 255]   # Blue: connected endpoints
        else:
            color_img[r0:r1, c0:c1] = [255, 0, 0]   # Red: unconnected endpoints

    out_vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
    cv2.imwrite(out_vis_path, color_img)
    logger.info(f"可视化图已保存：{out_vis_path}")

    # --- Output 3: split the triptych into three separate images ---
    # Image 1: original skeleton image (with endpoint markers, consistent with the left panel of the triptych)
    ep_array = np.array(endpoints)
    fig_single_1, ax_single_1 = plt.subplots(figsize=(6, 6))
    ax_single_1.imshow(original_skeleton, cmap='gray')
    ax_single_1.set_title("原始骨架图", fontfamily='SimHei')
    ax_single_1.axis('off')
    if len(ep_array) > 0:
        ax_single_1.scatter(ep_array[:, 1], ep_array[:, 0],
                            c='red', s=10, marker='o', label='断点')
        ax_single_1.legend(loc='upper right', fontsize=8)
    out_split_1 = os.path.join(output_dir, f"{base_name}_result_1_original.png")
    fig_single_1.savefig(out_split_1, dpi=150, bbox_inches='tight')
    plt.close(fig_single_1)
    logger.info(f"拆分图1已保存：{out_split_1}")

    # Image 2: colour reconstruction visualisation (consistent with the middle panel of the triptych)
    out_split_2 = os.path.join(output_dir, f"{base_name}_result_2_visualization.png")
    cv2.imwrite(out_split_2, color_img)
    logger.info(f"拆分图2已保存：{out_split_2}")

    # Image 3: reconstructed binary skeleton image (consistent with the right panel of the triptych)
    out_split_3 = os.path.join(output_dir, f"{base_name}_result_3_reconstructed.png")
    cv2.imwrite(out_split_3, reconstructed_skeleton)
    logger.info(f"拆分图3已保存：{out_split_3}")

    # --- Matplotlib comprehensive visualisation ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("基于ACO的裂隙骨架重构结果", fontsize=14, fontfamily='SimHei')

    axes[0].imshow(original_skeleton, cmap='gray')
    axes[0].set_title("原始骨架图", fontfamily='SimHei')
    axes[0].axis('off')

    # Mark endpoints on the original image
    if len(ep_array) > 0:
        axes[0].scatter(ep_array[:, 1], ep_array[:, 0],
                        c='red', s=10, marker='o', label='断点')
        axes[0].legend(loc='upper right', fontsize=8)

    axes[1].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("重构结果（绿=新连接，红=未连端点，蓝=已连端点）",
                       fontfamily='SimHei', fontsize=9)
    axes[1].axis('off')

    axes[2].imshow(reconstructed_skeleton, cmap='gray')
    axes[2].set_title("重构后骨架图", fontfamily='SimHei')
    axes[2].axis('off')

    plt.tight_layout()
    out_fig_path = os.path.join(output_dir, f"{base_name}_result.png")
    plt.savefig(out_fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"综合结果图已保存：{out_fig_path}")

    # --- ACO convergence curve ---
    if aco_score_history and len(aco_score_history) > 1:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(range(1, len(aco_score_history) + 1), aco_score_history,
                 'b-o', markersize=3, linewidth=1.5)
        ax2.set_xlabel("迭代次数")
        ax2.set_ylabel("最优方案得分")
        ax2.set_title("ACO优化收敛曲线", fontfamily='SimHei')
        ax2.grid(True, alpha=0.3)
        out_conv_path = os.path.join(output_dir, f"{base_name}_convergence.png")
        plt.tight_layout()
        plt.savefig(out_conv_path, dpi=120, bbox_inches='tight')
        plt.close()
        logger.info(f"收敛曲线已保存：{out_conv_path}")

def run_crack_aco_reconstruction(
    image_path: str,
    output_dir: str = "results",
    params: Optional[dict] = None
) -> Tuple[np.ndarray, List[Dict]]:

    if params is None:
        params = DEFAULT_PARAMS.copy()
    else:
        # Merge default parameters
        merged = DEFAULT_PARAMS.copy()
        merged.update(params)
        params = merged

    logger.info("=" * 60)
    logger.info("裂隙骨架ACO重构程序启动")
    logger.info("=" * 60)

    # --- 1. Loading and preprocessing ---
    skeleton = load_and_preprocess_image(
        image_path,
        binarize_threshold=params["binarize_threshold"],
        do_skeletonize=params["do_skeletonize"]
    )

    # --- 2. Connected component analysis ---
    labeled_map, num_components = extract_connected_components(skeleton)

    # --- 3. Endpoint detection ---
    endpoints, branchpoints = detect_endpoints_and_branchpoints(skeleton)

    if len(endpoints) == 0:
        logger.warning("未检测到断点，骨架可能已经完整连续，程序退出")
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_reconstructed.png"), skeleton)
        return skeleton, []

    # --- 4. Build the endpoint label dictionary ---
    endpoint_labels: Dict[Tuple[int, int], int] = {}
    for ep in endpoints:
        r, c = ep
        endpoint_labels[ep] = int(labeled_map[r, c])

    # --- 5. Local direction estimation ---
    logger.info(f"估计 {len(endpoints)} 个断点的局部方向...")
    directions: Dict[Tuple[int, int], np.ndarray] = {}
    for ep in endpoints:
        directions[ep] = estimate_endpoint_direction(
            skeleton, ep,
            direction_window=params["direction_window"]
        )

    # --- 6. Generate candidate connection pairs ---
    candidates = generate_candidate_pairs(
        skeleton, endpoints, directions, endpoint_labels, labeled_map, params
    )

    if not candidates:
        logger.warning("没有满足条件的候选连接对，程序退出")
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_reconstructed.png"), skeleton)
        return skeleton, []

    # --- 7. ACO optimisation ---
    aco = CrackSkeletonACOReconstructor(candidates, params)
    best_connections, best_score = aco.run()

    logger.info(f"\n最终选中 {len(best_connections)} 条连接：")
    for i, conn in enumerate(best_connections):
        logger.info(
            f"  [{i+1}] {conn['ep1']} <-> {conn['ep2']}，"
            f"距离={conn['dist']:.1f}px，"
            f"评分={conn['score']:.3f}，"
            f"连通域：{conn['label1']} <-> {conn['label2']}"
        )

    # --- 8. Skeleton reconstruction ---
    use_smooth = params.get("use_smooth_curve", False)
    reconstructed_skeleton, all_new_pixels = reconstruct_skeleton(
        skeleton, best_connections, directions, use_smooth_curve=use_smooth
    )

    # --- 9. Visualisation output ---
    visualize_results(
        original_skeleton=skeleton,
        reconstructed_skeleton=reconstructed_skeleton,
        endpoints=endpoints,
        best_connections=best_connections,
        all_new_pixels=all_new_pixels,
        output_dir=output_dir,
        input_filename=image_path,
        aco_score_history=aco.score_history
    )

    logger.info("=" * 60)
    logger.info("重构完成！")
    logger.info("=" * 60)

    return reconstructed_skeleton, best_connections

RUN_CONFIG = {
    # [Required] Input skeleton image path (supports png / jpg / bmp)
    "input_image": r"E:\Project(XHB)\Ant Colony Algorithm\1.jpeg",

    # [Required] Result output directory (created automatically if it does not exist)
    "output_dir": r"results(100-10-50-50)111",

    # Random seed (ensures reproducibility)
    "random_seed": 42,

    # Algorithm parameters (override DEFAULT_PARAMS; leave as {} to use all default values)
    "params": {
        "binarize_threshold": 50,     # Binarisation threshold (0-255)
        "do_skeletonize": True,        # Whether to thin the input again
        "direction_window": 10,        # Number of traced pixels for local direction estimation
        "max_gap": 150,                 # Maximum connection distance (pixels) modified
        "min_gap": 10,                  # Minimum connection distance (pixels) modified
        "max_angle_deg": 60,           # Direction angle tolerance (degrees)
        "ant_count": 30,               # Number of ants modified
        "max_iter": 50,                # Number of iterations modified
        "alpha": 1.0,                  # Pheromone weight
        "beta": 2.0,                   # Heuristic weight
        "rho": 0.1,                    # Pheromone evaporation rate
        "w_dist": 0.25,                # Weight of the distance score
        "w_dir": 0.35,                 # Weight of directional consistency
        "w_smooth": 0.20,              # Smoothness weight
        "w_clear": 0.10,               # Weight of the interference penalty
        "w_topo": 0.10,                # Weight of topological rationality
        "use_smooth_curve": False,     # True = Bezier curve, False = straight line
    },
}


def _validate_run_config(run_config: dict):
    input_image = run_config.get("input_image", "")
    if not input_image:
        raise ValueError(
            "RUN_CONFIG['input_image'] 不能为空，请填写真实输入图像路径。"
        )
    if not os.path.exists(input_image):
        raise FileNotFoundError(f"找不到输入图像：{input_image}")

    allowed_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ext = os.path.splitext(input_image)[1].lower()
    if ext not in allowed_ext:
        raise ValueError(f"输入文件扩展名不受支持：{ext}，请使用常见图像格式。")


if __name__ == "__main__":
    _validate_run_config(RUN_CONFIG)

    # Fix the random seed to ensure reproducibility
    random_seed = int(RUN_CONFIG.get("random_seed", 42))
    random.seed(random_seed)
    np.random.seed(random_seed)

    input_image = RUN_CONFIG["input_image"]
    output_dir = RUN_CONFIG.get("output_dir", "results")
    custom_params = RUN_CONFIG.get("params", {})

    logger.info(f"当前输入图像：{input_image}")
    logger.info(f"当前输出目录：{output_dir}")

    run_crack_aco_reconstruction(
        image_path=input_image,
        output_dir=output_dir,
        params=custom_params if custom_params else None
    )
