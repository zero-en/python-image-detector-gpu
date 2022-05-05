import numpy as np


def non_max_suppression(likelihoods, overlap_thresh: float):
    """NMS法を使って重なっている検出矩形の中からスコアが低いものを削減する"""
    rects = np.asarray([l[0] for l in likelihoods])
    scores = np.asarray([l[1] for l in likelihoods])
    if rects.size == 0:
        return []

    rects = rects.astype("float")
    x1, y1, x2, y2 = np.hsplit(rects, [1, 2, 3])

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.argsort(scores)
    selected = []

    while len(indices) > 0:
        last = len(indices) - 1

        selected_index = indices[last]
        remaining_indices = indices[:last]
        selected.append(selected_index)

        i_x1 = np.maximum(x1[selected_index], x1[remaining_indices])
        i_y1 = np.maximum(y1[selected_index], y1[remaining_indices])
        i_x2 = np.minimum(x2[selected_index], x2[remaining_indices])
        i_y2 = np.minimum(y2[selected_index], y2[remaining_indices])

        i_w = np.maximum(0, i_x2 - i_x1 + 1)
        i_h = np.maximum(0, i_y2 - i_y1 + 1)

        overlap = (i_w * i_h) / area[remaining_indices]
        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return [(tuple(rect), score) for rect, score in
            zip(rects[selected].tolist(), scores[selected].tolist())]
