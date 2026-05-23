import heapq
import math

import numpy as np
from skimage.draw import polygon as draw_polygon
from skimage.morphology import skeletonize


def compute_skeleton_farthest_endpoints(points, padding=8):
    # 输入是轮廓坐标点，先栅格化成局部二值区域，再做骨架和最远端点搜索。
    if points is None:
        return None

    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return None

    mask_result = _polygon_to_local_mask(points, padding=padding)
    if mask_result is None:
        return None

    mask, offset = mask_result
    skeleton = _skeletonize(mask)
    skeleton_pixels = np.argwhere(skeleton > 0)
    if skeleton_pixels.size == 0:
        return None

    endpoints = _find_skeleton_endpoints(skeleton)
    endpoint_count = len(endpoints)

    if endpoint_count >= 2:
        p1, p2, distance = _find_farthest_endpoint_pair(skeleton, endpoints)
    else:
        p1, p2, distance = _find_farthest_skeleton_pair(skeleton, skeleton_pixels)

    if p1 is None or p2 is None:
        return None

    p1_global = (int(p1[1] + offset[0]), int(p1[0] + offset[1]))
    p2_global = (int(p2[1] + offset[0]), int(p2[0] + offset[1]))

    return {
        "method": "skeleton",
        "p1": p1_global,
        "p2": p2_global,
        "skeleton_pixels": _to_global_pixels(skeleton, offset),
        "center_pt": (
            int(round((p1_global[0] + p2_global[0]) / 2.0)),
            int(round((p1_global[1] + p2_global[1]) / 2.0)),
        ),
        "endpoint_count": endpoint_count,
        "path_length": float(distance),
    }


def _polygon_to_local_mask(points, padding=8):
    min_xy = np.floor(np.min(points, axis=0)).astype(int) - padding
    max_xy = np.ceil(np.max(points, axis=0)).astype(int) + padding
    width = int(max_xy[0] - min_xy[0] + 1)
    height = int(max_xy[1] - min_xy[1] + 1)
    if width <= 0 or height <= 0:
        return None

    shifted = np.round(points - min_xy).astype(np.int32)
    rr, cc = draw_polygon(shifted[:, 1], shifted[:, 0], shape=(height, width))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rr, cc] = 1
    return mask, (int(min_xy[0]), int(min_xy[1]))


def _skeletonize(mask):
    return skeletonize(mask > 0).astype(np.uint8)


def _find_skeleton_endpoints(skeleton):
    skeleton_binary = (skeleton > 0).astype(np.uint8)
    kernel = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        dtype=np.uint8
    )
    neighbor_count = _convolve2d_same(skeleton_binary, kernel)
    endpoint_mask = (skeleton_binary == 1) & (neighbor_count == 1)
    return [tuple(pixel) for pixel in np.argwhere(endpoint_mask)]


def _find_farthest_endpoint_pair(skeleton, endpoints):
    best_pair = (None, None)
    best_distance = -1.0

    for start in endpoints:
        distances = _dijkstra_on_skeleton(skeleton, start)
        for end in endpoints:
            distance = distances.get(end, -1.0)
            if distance > best_distance:
                best_distance = distance
                best_pair = (start, end)

    return best_pair[0], best_pair[1], best_distance


def _find_farthest_skeleton_pair(skeleton, skeleton_pixels):
    start = tuple(skeleton_pixels[0])
    first_pass = _dijkstra_on_skeleton(skeleton, start)
    if not first_pass:
        return None, None, -1.0

    p1 = max(first_pass, key=first_pass.get)
    second_pass = _dijkstra_on_skeleton(skeleton, p1)
    if not second_pass:
        return None, None, -1.0

    p2 = max(second_pass, key=second_pass.get)
    return p1, p2, second_pass[p2]


def _dijkstra_on_skeleton(skeleton, start):
    skeleton_binary = skeleton > 0
    if not skeleton_binary[start]:
        return {}

    distances = {start: 0.0}
    queue = [(0.0, start)]

    while queue:
        current_distance, current = heapq.heappop(queue)
        if current_distance > distances[current]:
            continue

        for neighbor, weight in _iter_skeleton_neighbors(skeleton_binary, current):
            next_distance = current_distance + weight
            if next_distance < distances.get(neighbor, math.inf):
                distances[neighbor] = next_distance
                heapq.heappush(queue, (next_distance, neighbor))

    return distances


def _iter_skeleton_neighbors(skeleton_binary, pixel):
    row, col = pixel
    height, width = skeleton_binary.shape

    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row == 0 and d_col == 0:
                continue

            next_row = row + d_row
            next_col = col + d_col
            if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                continue
            if not skeleton_binary[next_row, next_col]:
                continue

            weight = math.sqrt(2.0) if d_row != 0 and d_col != 0 else 1.0
            yield (next_row, next_col), weight


def _convolve2d_same(image, kernel):
    # 只用于 3x3 端点邻域计数，避免再引入额外依赖。
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
    out = np.zeros_like(image, dtype=np.uint8)

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            window = padded[row : row + kernel.shape[0], col : col + kernel.shape[1]]
            out[row, col] = int(np.sum(window * kernel))

    return out


def _to_global_pixels(skeleton, offset):
    pixels = np.argwhere(skeleton > 0)
    return [
        (int(col + offset[0]), int(row + offset[1]))
        for row, col in pixels
    ]
