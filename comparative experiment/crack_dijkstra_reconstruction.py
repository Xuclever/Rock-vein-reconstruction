import os
import math
import time
import random
import logging
import warnings
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    # ---------- Preprocessing parameters ----------
    "binarize_threshold": 128,         # Binarization threshold (0-255); higher values retain fewer white pixels
    "do_skeletonize": True,            # Whether to perform skeleton thinning first (set to False if the input is already thinned)

    # ---------- Endpoint direction estimation ----------
    "direction_window": 10,            # Number of pixels traced inward from the endpoint; larger values produce smoother directions

    # ---------- Candidate edge filtering ----------
    "max_gap": 60,                     # Maximum connection distance (pixels)
    "min_gap": 3,                      # Minimum connection distance (pixels)
    "max_angle_deg": 60,               # Maximum allowed angle with the endpoint direction (degrees)
    "max_candidates_per_endpoint": 8,  # Maximum number of candidate connections retained per endpoint

    # ---------- Candidate edge scoring weights ----------
    "w_dist": 0.25,                    # Distance term weight
    "w_dir": 0.35,                     # Direction consistency weight
    "w_smooth": 0.20,                  # Smoothness weight
    "w_clear": 0.10,                   # Interference penalty weight (stray skeleton pixels near the line)
    "w_topo": 0.10,                    # Topological rationality weight

    # ---------- Scoring hyperparameters ----------
    "sigma_d": 20.0,                   # Distance decay coefficient (larger values penalise long distances less)
    "sigma_theta": 30.0,               # Angle decay coefficient (larger values are more tolerant)
    "interference_buffer": 3,          # Interference detection buffer radius (pixels)

    # ---------- Dijkstra optimisation parameters ----------
    "max_path_hops": 2,                # Maximum number of candidate edges allowed in the shortest path
    "max_path_cost": 2.5,              # Upper limit of total shortest-path cost (smaller values are more conservative)
    "cost_lambda_dist": 0.15,          # Distance penalty coefficient in the edge cost
    "path_length_penalty": 0.10,       # Additional penalty for multi-hop paths (added for each extra hop)

    # ---------- Connection-line drawing ----------
    "use_smooth_curve": False,         # Reserved switch: the current implementation uses straight-line connections by default
}

def _bresenham_line(p1: Tuple[int, int], p2: Tuple[int, int]) -> List[Tuple[int, int]]:
    r1, c1 = p1
    r2, c2 = p2
    points = []
    dr, dc = abs(r2 - r1), abs(c2 - c1)
    sr = 1 if r2 > r1 else -1
    sc = 1 if c2 > c1 else -1
    err = dr - dc
    r, c = r1, c1
    while True:
        points.append((r, c))
        if r == r2 and c == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return points


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p, q, r) -> bool:
    return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])


def segments_intersect(
    p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int], p4: Tuple[int, int]
) -> bool:
    if p1 in (p3, p4) or p2 in (p3, p4):
        return False
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    if d1 == 0 and _on_segment(p3, p1, p4):
        return True
    if d2 == 0 and _on_segment(p3, p2, p4):
        return True
    if d3 == 0 and _on_segment(p1, p3, p2):
        return True
    if d4 == 0 and _on_segment(p1, p4, p2):
        return True
    return False


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    cosv = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
    return math.degrees(math.acos(cosv))

def load_and_preprocess_image(image_path: str, threshold: int, do_skeleton: bool) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到输入文件：{image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")

    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    if do_skeleton:
        skel = skeletonize(binary > 0)
        return (skel * 255).astype(np.uint8)
    return binary.astype(np.uint8)


def extract_connected_components(skeleton: np.ndarray) -> Tuple[np.ndarray, int]:
    binary = (skeleton > 0).astype(np.uint8)
    num_labels, labeled = cv2.connectedComponents(binary, connectivity=8)
    return labeled, num_labels - 1


def detect_endpoints_and_branchpoints(
    skeleton: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    binary = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    ncount = cv2.filter2D(binary, -1, kernel.astype(np.float32)).astype(np.int32)
    mask = binary > 0
    endpoints = list(zip(*np.where(mask & (ncount == 1))))
    branchpoints = list(zip(*np.where(mask & (ncount >= 3))))
    return endpoints, branchpoints


def _trace_skeleton_path(skeleton: np.ndarray, start: Tuple[int, int], max_steps: int) -> List[Tuple[int, int]]:
    binary = skeleton > 0
    h, w = binary.shape
    path = [start]
    visited = {start}
    cur = start
    for _ in range(max_steps):
        r, c = cur
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and binary[nr, nc] and (nr, nc) not in visited:
                    nbrs.append((nr, nc))
        if not nbrs:
            break
        cur = nbrs[0]
        visited.add(cur)
        path.append(cur)
    return path


def estimate_endpoint_direction(skeleton: np.ndarray, endpoint: Tuple[int, int], direction_window: int) -> np.ndarray:
    path = _trace_skeleton_path(skeleton, endpoint, max_steps=direction_window)
    if len(path) < 2:
        return np.array([1.0, 0.0])
    pts = np.array(path, dtype=np.float64)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, np.argmax(eigvals)]
    inner = np.array(path[-1], dtype=np.float64) - np.array(path[0], dtype=np.float64)
    if np.dot(vec, inner) < 0:
        vec = -vec
    vec = -vec
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-8 else np.array([1.0, 0.0])

def line_interference_penalty(
    skeleton: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    buffer_width: int,
    label1: int,
    label2: int,
    labeled_map: np.ndarray,
) -> float:
    h, w = skeleton.shape
    line = _bresenham_line(p1, p2)
    if not line:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    for r, c in line:
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * buffer_width + 1, 2 * buffer_width + 1))
    buf = cv2.dilate(mask, k)
    sk = (skeleton > 0).astype(np.uint8)
    inter = (buf & sk).astype(bool)
    own = (labeled_map == label1) | (labeled_map == label2)
    inter = inter & (~own)
    cnt = float(np.sum(inter))
    denom = max(len(line) * max(buffer_width, 1) * 2, 1)
    return float(min(cnt / denom, 1.0))


def evaluate_connection_pair(
    skeleton: np.ndarray,
    ep1: Tuple[int, int],
    ep2: Tuple[int, int],
    dir1: np.ndarray,
    dir2: np.ndarray,
    label1: int,
    label2: int,
    labeled_map: np.ndarray,
    params: dict,
) -> float:
    p1, p2 = np.array(ep1, dtype=np.float64), np.array(ep2, dtype=np.float64)
    vec = p2 - p1
    dist = float(np.linalg.norm(vec))
    if dist < 1e-8:
        return 0.0
    conn = vec / dist

    s_dist = math.exp(-dist / float(params["sigma_d"]))
    th1 = min(angle_between(dir1, conn), 180.0 - angle_between(dir1, conn))
    th2 = min(angle_between(dir2, -conn), 180.0 - angle_between(dir2, -conn))
    s_dir = math.exp(-(th1 + th2) / (2.0 * float(params["sigma_theta"])))
    s_smooth = max(0.0, 1.0 - ((th1 + th2) / 2.0) / float(params["max_angle_deg"]))

    clear_pen = line_interference_penalty(
        skeleton, ep1, ep2, int(params["interference_buffer"]), label1, label2, labeled_map
    )
    s_clear = 1.0 - clear_pen

    s_topo = 1.0 if label1 != label2 else max(0.2, 1.0 - math.exp(-dist / (2.0 * params["sigma_d"])))

    score = (
        params["w_dist"] * s_dist
        + params["w_dir"] * s_dir
        + params["w_smooth"] * s_smooth
        + params["w_clear"] * s_clear
        + params["w_topo"] * s_topo
    )
    return float(np.clip(score, 0.0, 1.0))


def generate_candidate_pairs(
    skeleton: np.ndarray,
    endpoints: List[Tuple[int, int]],
    directions: Dict[Tuple[int, int], np.ndarray],
    endpoint_labels: Dict[Tuple[int, int], int],
    labeled_map: np.ndarray,
    params: dict,
) -> List[Dict]:
    max_gap = float(params["max_gap"])
    min_gap = float(params["min_gap"])
    max_angle = float(params["max_angle_deg"])
    max_per_ep = int(params["max_candidates_per_endpoint"])

    n = len(endpoints)
    if n == 0:
        return []
    ep_arr = np.array(endpoints, dtype=np.float64)
    counts = [0] * n
    candidates: List[Dict] = []

    for i in range(n):
        if counts[i] >= max_per_ep:
            continue
        dists = np.linalg.norm(ep_arr - ep_arr[i], axis=1)
        order = np.argsort(dists)
        added = 0
        ep1 = endpoints[i]
        dir1 = directions[ep1]
        label1 = endpoint_labels[ep1]
        for j in order:
            j = int(j)
            if j <= i or counts[j] >= max_per_ep:
                continue
            d = float(dists[j])
            if d < min_gap:
                continue
            if d > max_gap:
                break
            ep2 = endpoints[j]
            dir2 = directions[ep2]
            conn = np.array([ep2[0] - ep1[0], ep2[1] - ep1[1]], dtype=np.float64)
            conn = conn / (np.linalg.norm(conn) + 1e-8)
            t1 = min(angle_between(dir1, conn), 180.0 - angle_between(dir1, conn))
            t2 = min(angle_between(dir2, -conn), 180.0 - angle_between(dir2, -conn))
            if t1 > max_angle or t2 > max_angle:
                continue
            label2 = endpoint_labels[ep2]
            score = evaluate_connection_pair(
                skeleton, ep1, ep2, dir1, dir2, label1, label2, labeled_map, params
            )
            if score <= 0.01:
                continue
            candidates.append(
                {
                    "ep1": ep1,
                    "ep2": ep2,
                    "dist": d,
                    "score": score,
                    "label1": label1,
                    "label2": label2,
                    "idx1": i,
                    "idx2": j,
                }
            )
            counts[i] += 1
            counts[j] += 1
            added += 1
            if added >= max_per_ep:
                break

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

class CrackSkeletonDijkstraReconstructor:
    def __init__(self, candidates: List[Dict], params: dict):
        self.candidates = candidates
        self.params = params
        self.best_solution: List[int] = []
        self.best_score: float = 0.0
        self.score_history: List[float] = []
        self.max_path_hops = int(params["max_path_hops"])
        self.max_path_cost = float(params["max_path_cost"])
        self.cost_lambda_dist = float(params["cost_lambda_dist"])
        self.path_length_penalty = float(params["path_length_penalty"])
        self.edge_index_map = {}
        for i, c in enumerate(candidates):
            self.edge_index_map[self._edge_key(c["ep1"], c["ep2"])] = i
        self.graph = self._build_graph()

    @staticmethod
    def _edge_key(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (a, b) if a <= b else (b, a)

    def _build_graph(self) -> nx.Graph:
        g = nx.Graph()
        mg = max(float(self.params["max_gap"]), 1.0)
        for i, c in enumerate(self.candidates):
            score = float(np.clip(c["score"], 1e-6, 1.0))
            dist_n = float(c["dist"]) / mg
            cost = (1.0 - score) + self.cost_lambda_dist * dist_n
            g.add_edge(c["ep1"], c["ep2"], weight=max(cost, 1e-6), edge_idx=i)
        return g

    def _path_to_edges(self, path_nodes: List[Tuple[int, int]]) -> List[int]:
        edges = []
        for i in range(len(path_nodes) - 1):
            k = self._edge_key(path_nodes[i], path_nodes[i + 1])
            if k not in self.edge_index_map:
                return []
            edges.append(self.edge_index_map[k])
        return edges

    def _check_crossing(self, idx1: int, idx2: int) -> bool:
        c1, c2 = self.candidates[idx1], self.candidates[idx2]
        return segments_intersect(c1["ep1"], c1["ep2"], c2["ep1"], c2["ep2"])

    def _enumerate_path_proposals(self) -> List[Dict]:
        proposals = []
        if len(self.candidates) == 0:
            return proposals
        seen = set()
        for source in self.graph.nodes:
            lengths, paths = nx.single_source_dijkstra(
                self.graph, source=source, cutoff=self.max_path_cost, weight="weight"
            )
            for target, path_cost in lengths.items():
                if target == source:
                    continue
                pair = self._edge_key(source, target)
                if pair in seen:
                    continue
                nodes = paths.get(target, [])
                if len(nodes) < 2:
                    continue
                edge_indices = self._path_to_edges(nodes)
                if not edge_indices or len(edge_indices) > self.max_path_hops:
                    continue
                score_sum = float(sum(self.candidates[e]["score"] for e in edge_indices))
                score = score_sum - float(path_cost) - self.path_length_penalty * max(0, len(edge_indices) - 1)
                proposals.append(
                    {
                        "path_nodes": nodes,
                        "edge_indices": edge_indices,
                        "path_cost": float(path_cost),
                        "path_score": float(score),
                    }
                )
                seen.add(pair)
        proposals.sort(key=lambda x: (-x["path_score"], x["path_cost"]))
        return proposals

    def _select_non_conflict_paths(self, proposals: List[Dict]) -> Tuple[List[int], float]:
        selected, used_edges = [], set()
        used_nodes: Set[Tuple[int, int]] = set()
        total = 0.0
        for p in proposals:
            if p["path_score"] <= 0:
                continue
            nodes = p["path_nodes"]
            edges = p["edge_indices"]
            if any(n in used_nodes for n in nodes):
                continue
            bad = False
            for e in edges:
                if e in used_edges:
                    bad = True
                    break
                for se in selected:
                    if self._check_crossing(e, se):
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            for e in edges:
                selected.append(e)
                used_edges.add(e)
                total += float(self.candidates[e]["score"])
            for n in nodes:
                used_nodes.add(n)
            self.score_history.append(total)
        return selected, total

    def run(self) -> Tuple[List[Dict], float]:
        if len(self.candidates) == 0:
            return [], 0.0
        t0 = time.time()
        proposals = self._enumerate_path_proposals()
        self.best_solution, self.best_score = self._select_non_conflict_paths(proposals)
        logger.info(
            f"Dijkstra完成: 候选边={len(self.candidates)}, 路径候选={len(proposals)}, "
            f"选中边={len(self.best_solution)}, 最优分={self.best_score:.4f}, 耗时={time.time()-t0:.2f}s"
        )
        return [self.candidates[i] for i in self.best_solution], self.best_score

def draw_connection(
    skeleton: np.ndarray,
    ep1: Tuple[int, int],
    ep2: Tuple[int, int],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    out = skeleton.copy()
    pts = _bresenham_line(ep1, ep2)
    h, w = out.shape
    for r, c in pts:
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = 255
    return out, pts


def reconstruct_skeleton(
    skeleton: np.ndarray, best_connections: List[Dict]
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
    out = skeleton.copy()
    all_new_pixels = []
    for c in best_connections:
        out, pts = draw_connection(out, c["ep1"], c["ep2"])
        all_new_pixels.append(pts)
    return out, all_new_pixels


def visualize_results(
    original_skeleton: np.ndarray,
    reconstructed_skeleton: np.ndarray,
    endpoints: List[Tuple[int, int]],
    best_connections: List[Dict],
    all_new_pixels: List[List[Tuple[int, int]]],
    output_dir: str,
    input_filename: str,
    optimization_score_history: Optional[List[float]] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_filename))[0]

    out_reconstructed = os.path.join(output_dir, f"{base}_reconstructed.png")
    cv2.imwrite(out_reconstructed, reconstructed_skeleton)

    h, w = original_skeleton.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[original_skeleton > 0] = [200, 200, 200]
    for pts in all_new_pixels:
        for r, c in pts:
            if 0 <= r < h and 0 <= c < w:
                color[r, c] = [0, 255, 0]

    linked = set([c["ep1"] for c in best_connections] + [c["ep2"] for c in best_connections])
    for ep in endpoints:
        r, c = ep
        r0, r1 = max(0, r - 2), min(h, r + 3)
        c0, c1 = max(0, c - 2), min(w, c + 3)
        color[r0:r1, c0:c1] = [0, 0, 255] if ep in linked else [255, 0, 0]

    out_vis = os.path.join(output_dir, f"{base}_visualization.png")
    cv2.imwrite(out_vis, color)

    # Triptych figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("基于Dijkstra的裂隙骨架重构结果", fontsize=14, fontfamily="SimHei")
    axes[0].imshow(original_skeleton, cmap="gray")
    axes[0].set_title("原始骨架图", fontfamily="SimHei")
    axes[0].axis("off")
    ep_arr = np.array(endpoints)
    if len(ep_arr) > 0:
        axes[0].scatter(ep_arr[:, 1], ep_arr[:, 0], c="red", s=10)
    axes[1].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    axes[1].set_title("彩色重构图", fontfamily="SimHei")
    axes[1].axis("off")
    axes[2].imshow(reconstructed_skeleton, cmap="gray")
    axes[2].set_title("重构骨架图", fontfamily="SimHei")
    axes[2].axis("off")
    plt.tight_layout()
    out_triplet = os.path.join(output_dir, f"{base}_result.png")
    plt.savefig(out_triplet, dpi=150, bbox_inches="tight")
    plt.close()

    # Save the triptych figure as separate images
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(original_skeleton, cmap="gray")
    ax1.set_title("原始骨架图", fontfamily="SimHei")
    ax1.axis("off")
    if len(ep_arr) > 0:
        ax1.scatter(ep_arr[:, 1], ep_arr[:, 0], c="red", s=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_result_1_original.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    cv2.imwrite(os.path.join(output_dir, f"{base}_result_2_visualization.png"), color)
    cv2.imwrite(os.path.join(output_dir, f"{base}_result_3_reconstructed.png"), reconstructed_skeleton)

    if optimization_score_history and len(optimization_score_history) > 1:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(range(1, len(optimization_score_history) + 1), optimization_score_history, "b-o", markersize=3)
        ax2.set_xlabel("选边步骤")
        ax2.set_ylabel("累计方案得分")
        ax2.set_title("Dijkstra选边累计得分曲线", fontfamily="SimHei")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base}_convergence.png"), dpi=120, bbox_inches="tight")
        plt.close(fig2)

    logger.info(f"结果已保存到: {output_dir}")

def run_crack_dijkstra_reconstruction(image_path: str, output_dir: str, params: Optional[dict] = None):
    cfg = DEFAULT_PARAMS.copy()
    if params:
        cfg.update(params)

    logger.info("=" * 60)
    logger.info("裂隙骨架 Dijkstra 重构启动")
    logger.info("=" * 60)

    skeleton = load_and_preprocess_image(image_path, cfg["binarize_threshold"], cfg["do_skeletonize"])
    labeled_map, n_comp = extract_connected_components(skeleton)
    endpoints, branchpoints = detect_endpoints_and_branchpoints(skeleton)
    logger.info(f"连通域: {n_comp}, 端点: {len(endpoints)}, 分叉点: {len(branchpoints)}")
    if len(endpoints) == 0:
        raise RuntimeError("未检测到断点，无法进行断裂连接。")

    endpoint_labels = {ep: int(labeled_map[ep[0], ep[1]]) for ep in endpoints}
    directions = {
        ep: estimate_endpoint_direction(skeleton, ep, cfg["direction_window"]) for ep in endpoints
    }

    candidates = generate_candidate_pairs(skeleton, endpoints, directions, endpoint_labels, labeled_map, cfg)
    logger.info(f"候选连接对数量: {len(candidates)}")
    if not candidates:
        raise RuntimeError("没有满足条件的候选连接对。")

    solver = CrackSkeletonDijkstraReconstructor(candidates, cfg)
    best_connections, best_score = solver.run()
    logger.info(f"最优方案得分: {best_score:.4f}, 连接数: {len(best_connections)}")

    reconstructed, all_new_pixels = reconstruct_skeleton(skeleton, best_connections)
    visualize_results(
        original_skeleton=skeleton,
        reconstructed_skeleton=reconstructed,
        endpoints=endpoints,
        best_connections=best_connections,
        all_new_pixels=all_new_pixels,
        output_dir=output_dir,
        input_filename=image_path,
        optimization_score_history=solver.score_history,
    )
    return reconstructed, best_connections

RUN_CONFIG = {
    "input_image": r"E:\Project(XHB)\Ant Colony Algorithm\1.jpeg",              # Required: absolute path of the input image
    "output_dir": r"results_dijkstra(150-10-4)",  # Directory for saving results
    "random_seed": 42,               # Random seed (fixed for reproducible results)
    "params": {
        "binarize_threshold": 50,   # Increase the threshold to reduce noise
        "do_skeletonize": True,     # Disable this if the input is already a one-pixel-wide skeleton
        "max_gap": 150,               # Allow a larger fracture connection distance
        "min_gap": 10,               # Minimum connection distance (too close is not connected, as it may already be adjacent)
        "max_angle_deg": 60,         # Stricter direction constraint
        "max_candidates_per_endpoint": 4,# Maximum number of candidate connections retained per endpoint. Larger values: more candidate edges and a larger search space, which may find better connections;
                                                                    # Smaller values: faster and more stable results, but real connections may be missed
                # ---------- Candidate edge scoring weights ----------
        "w_dist": 0.25,                    # Distance term weight. Larger values give more importance to nearby connections
        "w_dir": 0.35,                     # Direction consistency weight. Larger values give more importance to directional consistency
        "w_smooth": 0.20,                  # Smoothness weight. Larger values give more importance to smooth connections
        "w_clear": 0.10,                   # Interference penalty weight (stray skeleton pixels near the line). Larger values give more importance to connections with less interference
        "w_topo": 0.10,                    # Topological rationality weight. Larger values give more importance to topologically reasonable connections
        # ---------- Scoring hyperparameters ----------
        "sigma_d": 20.0,                   # Distance decay coefficient (larger values penalise long distances less)
        "sigma_theta": 30.0,               # Angle decay coefficient (larger values are more tolerant)
        "interference_buffer": 3,          # Interference detection buffer radius (pixels)
        # ---------- Dijkstra optimisation parameters ----------
        "max_path_hops": 2,                # Maximum number of candidate edges allowed in the shortest path
        "max_path_cost": 2.5,              # Upper limit of total shortest-path cost (smaller values are more conservative)
        "cost_lambda_dist": 0.15,          # Distance penalty coefficient in the edge cost
        "path_length_penalty": 0.10,       # Additional penalty for multi-hop paths (added for each extra hop)
        # ---------- Connection-line drawing ----------
        "use_smooth_curve": False,         # Reserved switch: the current implementation uses straight-line connections by default
    },
}


def _validate_run_config(run_config: dict):
    input_image = run_config.get("input_image", "")
    if not input_image:
        raise ValueError("RUN_CONFIG['input_image'] 不能为空，请填写真实输入图路径。")
    if not os.path.exists(input_image):
        raise FileNotFoundError(f"找不到输入图像: {input_image}")
    ext = os.path.splitext(input_image)[1].lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise ValueError(f"不支持的输入格式: {ext}")


if __name__ == "__main__":
    _validate_run_config(RUN_CONFIG)
    seed = int(RUN_CONFIG.get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    input_image = RUN_CONFIG["input_image"]
    output_dir = RUN_CONFIG.get("output_dir", "results_dijkstra")
    custom_params = RUN_CONFIG.get("params", {})
    logger.info(f"当前输入图像: {input_image}")
    logger.info(f"当前输出目录: {output_dir}")
    run_crack_dijkstra_reconstruction(input_image, output_dir, custom_params)

