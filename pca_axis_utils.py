import cv2
import numpy as np

DEFAULT_ENDPOINT_REGION_RATIO = 0.15


def compute_major_axis(points, endpoint_region_ratio=DEFAULT_ENDPOINT_REGION_RATIO):
    # 保持旧接口不变，默认返回优化后的主轴端点。
    comparison = compute_major_axis_comparison(
        points,
        endpoint_region_ratio=endpoint_region_ratio
    )
    if comparison is None:
        return None
    return comparison["center_pt"], comparison["optimized_p1"], comparison["optimized_p2"]


def compute_major_axis_comparison(points, endpoint_region_ratio=DEFAULT_ENDPOINT_REGION_RATIO):
    # 同时计算最初主轴和优化后主轴，便于界面层对比显示。
    if points is None:
        return None

    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return None

    mean = np.empty((0))
    mean, eigenvectors = cv2.PCACompute(points, mean)
    center = mean[0]
    center_pt = tuple(np.round(center).astype(int))

    # 第一特征向量就是多边形的主方向。
    axis = _normalize_vector(eigenvectors[0])
    if axis is None:
        return None

    original_p1, original_p2 = _find_axis_endpoints_by_intersection(center, axis, points)
    optimized_p1, optimized_p2 = _find_axis_endpoints_by_projection(
        center, axis, points, endpoint_region_ratio
    )

    return {
        "center_pt": center_pt,
        "original_p1": original_p1,
        "original_p2": original_p2,
        "optimized_p1": optimized_p1,
        "optimized_p2": optimized_p2,
    }


def _find_axis_endpoints_by_projection(center, axis, points, endpoint_region_ratio):
    # 在主轴投影两端的 K% 区域内，分别选择距离质心最远的轮廓点。
    diffs = points - center
    parallel = diffs @ axis
    radial = np.linalg.norm(diffs, axis=1)

    p1_idx, p2_idx = _select_endpoint_indices_by_region(
        parallel,
        radial,
        endpoint_region_ratio
    )

    if p1_idx is None or p2_idx is None or p1_idx == p2_idx:
        # 兜底方案：如果区域法选不出稳定端点，则退回纯主轴投影最远点。
        p1_idx = int(np.argmin(parallel))
        p2_idx = int(np.argmax(parallel))

    # 优先返回真实轮廓点，而不是主轴与边界的理论交点。
    p1 = tuple(np.round(points[p1_idx]).astype(int))
    p2 = tuple(np.round(points[p2_idx]).astype(int))
    return p1, p2


def _find_axis_endpoints_by_intersection(center, axis, points):
    # 最初版本：优先求主轴与多边形边界的真实交点。
    intersections = _line_polygon_intersections(center, axis, points)
    if len(intersections) >= 2:
        ts = np.asarray([item[0] for item in intersections], dtype=np.float32)
        p1 = tuple(np.round(center + axis * ts.min()).astype(int))
        p2 = tuple(np.round(center + axis * ts.max()).astype(int))
        return p1, p2

    # 如果交点不足，则退回纯主轴投影最远点。
    proj = (points - center) @ axis
    p1 = tuple(np.round(center + axis * proj.min()).astype(int))
    p2 = tuple(np.round(center + axis * proj.max()).astype(int))
    return p1, p2


def _line_polygon_intersections(center, axis, points):
    # 将无限长的 PCA 轴与多边形每一条边求交。
    intersections = []
    point_count = len(points)

    for idx in range(point_count):
        seg_start = points[idx]
        seg_end = points[(idx + 1) % point_count]
        hit = _line_segment_intersection(center, axis, seg_start, seg_end)
        if hit is None:
            continue

        t, point = hit
        # 相邻边在顶点处可能产生数值上重复的交点，这里做去重。
        is_duplicate = any(
            abs(t - existing_t) < 1e-3 or np.linalg.norm(point - existing_point) < 1.5
            for existing_t, existing_point in intersections
        )
        if not is_duplicate:
            intersections.append((t, point))

    return intersections


def _line_segment_intersection(line_point, line_dir, seg_start, seg_end, eps=1e-6):
    # 求无限直线与一条线段的交点。
    seg_dir = seg_end - seg_start
    denom = _cross_2d(line_dir, seg_dir)
    if abs(denom) < eps:
        return None

    diff = seg_start - line_point
    t = _cross_2d(diff, seg_dir) / denom
    u = _cross_2d(diff, line_dir) / denom
    # u 在 [0, 1] 范围内表示交点落在线段上。
    if -eps <= u <= 1.0 + eps:
        point = line_point + t * line_dir
        return float(t), point
    return None


def _cross_2d(a, b):
    # 2D 叉积标量，用于判断平行和求交。
    return float(a[0] * b[1] - a[1] * b[0])


def _normalize_vector(vector, eps=1e-6):
    # 归一化主轴方向，避免投影尺度受向量长度影响。
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return None
    return vector / norm


def _select_endpoint_indices_by_region(parallel, radial, endpoint_region_ratio):
    # 在投影最小端和最大端附近各取一段区域，再找距离质心最远的轮廓点。
    if parallel.size == 0:
        return None, None

    region_ratio = float(np.clip(endpoint_region_ratio, 0.0, 0.5))
    min_proj = float(np.min(parallel))
    max_proj = float(np.max(parallel))
    proj_span = max_proj - min_proj
    if proj_span <= 1e-6:
        return None, None

    band = proj_span * region_ratio
    min_threshold = min_proj + band
    max_threshold = max_proj - band

    min_candidates = np.flatnonzero(parallel <= min_threshold)
    max_candidates = np.flatnonzero(parallel >= max_threshold)

    if min_candidates.size == 0 or max_candidates.size == 0:
        return None, None

    min_best = int(min_candidates[np.argmax(radial[min_candidates])])
    max_best = int(max_candidates[np.argmax(radial[max_candidates])])
    return min_best, max_best
