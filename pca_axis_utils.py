import cv2
import numpy as np

DEFAULT_ENDPOINT_REGION_RATIO = 0.15
DEFAULT_OUTLIER_SIGMA = 2.5
MIN_INLIER_COUNT = 8
MIN_INLIER_RATIO = 0.4


def compute_major_axis(
    points,
    endpoint_region_ratio=DEFAULT_ENDPOINT_REGION_RATIO,
    outlier_sigma=DEFAULT_OUTLIER_SIGMA,
):
    # 保持旧接口不变，默认返回优化后的主轴端点。
    comparison = compute_major_axis_comparison(
        points,
        endpoint_region_ratio=endpoint_region_ratio,
        outlier_sigma=outlier_sigma,
    )
    if comparison is None:
        return None
    return comparison["center_pt"], comparison["optimized_p1"], comparison["optimized_p2"]


def compute_major_axis_comparison(
    points,
    endpoint_region_ratio=DEFAULT_ENDPOINT_REGION_RATIO,
    outlier_sigma=DEFAULT_OUTLIER_SIGMA,
):
    # 先基于全部点求一个初始 PCA 轴，再过滤远离主轴的离群点后重算稳定主轴。
    if points is None:
        return None

    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return None

    initial_center, initial_axis = _compute_pca_axis(points)
    if initial_center is None or initial_axis is None:
        return None

    original_p1, original_p2 = _find_axis_endpoints_by_intersection(
        initial_center, initial_axis, points
    )

    inlier_mask = _build_inlier_mask(points, initial_center, initial_axis, outlier_sigma)
    inlier_points = points[inlier_mask]
    if inlier_points.shape[0] >= 2:
        refined_center, refined_axis = _compute_pca_axis(inlier_points)
    else:
        refined_center, refined_axis = None, None

    if refined_center is None or refined_axis is None:
        refined_center = initial_center
        refined_axis = initial_axis
        inlier_points = points

    optimized_p1, optimized_p2 = _find_axis_endpoints_by_projection(
        refined_center,
        refined_axis,
        inlier_points,
        endpoint_region_ratio,
    )

    return {
        "center_pt": tuple(np.round(refined_center).astype(int)),
        "original_p1": original_p1,
        "original_p2": original_p2,
        "optimized_p1": optimized_p1,
        "optimized_p2": optimized_p2,
        "inlier_count": int(inlier_points.shape[0]),
        "point_count": int(points.shape[0]),
    }


def _compute_pca_axis(points):
    mean = np.empty((0), dtype=np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean)
    if mean is None or eigenvectors is None or len(mean) == 0 or len(eigenvectors) == 0:
        return None, None

    center = mean[0]
    axis = _normalize_vector(eigenvectors[0])
    if axis is None:
        return None, None
    return center, axis


def _build_inlier_mask(points, center, axis, outlier_sigma):
    # 用点到 PCA 主轴的垂直距离做鲁棒过滤，减少异常轮廓尖刺对主轴方向的影响。
    diffs = points - center
    parallel = diffs @ axis
    projected = np.outer(parallel, axis)
    orthogonal = diffs - projected
    distances = np.linalg.norm(orthogonal, axis=1)

    if distances.size == 0:
        return np.zeros((0,), dtype=bool)

    median_distance = float(np.median(distances))
    abs_dev = np.abs(distances - median_distance)
    mad = float(np.median(abs_dev))

    if mad <= 1e-6:
        distance_threshold = max(median_distance * 1.5, 2.0)
    else:
        robust_sigma = 1.4826 * mad
        sigma = max(float(outlier_sigma), 1.0)
        distance_threshold = median_distance + sigma * robust_sigma

    mask = distances <= distance_threshold

    min_keep = max(MIN_INLIER_COUNT, int(np.ceil(points.shape[0] * MIN_INLIER_RATIO)))
    if int(np.count_nonzero(mask)) < min_keep:
        return np.ones(points.shape[0], dtype=bool)
    return mask


def _find_axis_endpoints_by_projection(center, axis, points, endpoint_region_ratio):
    # 在主轴投影两端的 K% 区域内，分别选择沿主轴最极端的真实轮廓点。
    diffs = points - center
    parallel = diffs @ axis
    radial = np.linalg.norm(diffs, axis=1)

    p1_idx, p2_idx = _select_endpoint_indices_by_region(
        parallel,
        radial,
        endpoint_region_ratio,
    )

    if p1_idx is None or p2_idx is None or p1_idx == p2_idx:
        p1_idx = int(np.argmin(parallel))
        p2_idx = int(np.argmax(parallel))

    p1 = tuple(np.round(points[p1_idx]).astype(int))
    p2 = tuple(np.round(points[p2_idx]).astype(int))
    return p1, p2


def _find_axis_endpoints_by_intersection(center, axis, points):
    # 原始版本：求主轴与轮廓边界交点，用于和优化结果做对比。
    intersections = _line_polygon_intersections(center, axis, points)
    if len(intersections) >= 2:
        ts = np.asarray([item[0] for item in intersections], dtype=np.float32)
        p1 = tuple(np.round(center + axis * ts.min()).astype(int))
        p2 = tuple(np.round(center + axis * ts.max()).astype(int))
        return p1, p2

    proj = (points - center) @ axis
    p1 = tuple(np.round(center + axis * proj.min()).astype(int))
    p2 = tuple(np.round(center + axis * proj.max()).astype(int))
    return p1, p2


def _line_polygon_intersections(center, axis, points):
    intersections = []
    point_count = len(points)

    for idx in range(point_count):
        seg_start = points[idx]
        seg_end = points[(idx + 1) % point_count]
        hit = _line_segment_intersection(center, axis, seg_start, seg_end)
        if hit is None:
            continue

        t, point = hit
        is_duplicate = any(
            abs(t - existing_t) < 1e-3 or np.linalg.norm(point - existing_point) < 1.5
            for existing_t, existing_point in intersections
        )
        if not is_duplicate:
            intersections.append((t, point))

    return intersections


def _line_segment_intersection(line_point, line_dir, seg_start, seg_end, eps=1e-6):
    seg_dir = seg_end - seg_start
    denom = _cross_2d(line_dir, seg_dir)
    if abs(denom) < eps:
        return None

    diff = seg_start - line_point
    t = _cross_2d(diff, seg_dir) / denom
    u = _cross_2d(diff, line_dir) / denom
    if -eps <= u <= 1.0 + eps:
        point = line_point + t * line_dir
        return float(t), point
    return None


def _cross_2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _normalize_vector(vector, eps=1e-6):
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return None
    return vector / norm


def _select_endpoint_indices_by_region(parallel, radial, endpoint_region_ratio):
    # 只在主轴投影两端的局部区域选点，减少中部宽轮廓对端点的干扰。
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

    min_best = int(min_candidates[np.argmin(parallel[min_candidates])])
    max_best = int(max_candidates[np.argmax(parallel[max_candidates])])

    if min_best == max_best:
        min_best = int(min_candidates[np.argmax(radial[min_candidates])])
        max_best = int(max_candidates[np.argmax(radial[max_candidates])])

    return min_best, max_best
