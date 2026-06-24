import math
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


def getDistance(x1, y1, x2, y2):
    return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))


def CLEAR_MOD_HUN(gt, det):
    """
    Compute CLEAR Detection metrics (MODA, MODP, recall, precision).

    Args:
        gt:  (N, 4) array — columns [frame, person_id, x, y]
        det: (M, 4) array — columns [frame, det_id, x, y]

    Returns:
        recall, precision, MODA, MODP
    """
    td = 50 / 2.5

    F = int(max(gt[:, 0])) + 1
    Ngt = int(max(gt[:, 1])) + 1

    M = np.zeros((F, Ngt))
    c = np.zeros((1, F))
    fp = np.zeros((1, F))
    m = np.zeros((1, F))
    g = np.zeros((1, F))
    distances = np.inf * np.ones((F, Ngt))

    for t in range(1, F + 1):
        GTsInFrame = np.where(gt[:, 0] == t - 1)[0]
        DetsInFrame = np.where(det[:, 0] == t - 1)[0]
        GTsInFrame = GTsInFrame.reshape(1, -1)
        DetsInFrame = DetsInFrame.reshape(1, -1)

        Ngtt = GTsInFrame.shape[1]
        Nt = DetsInFrame.shape[1]
        g[0, t - 1] = Ngtt

        if Ngtt > 0 and Nt > 0:
            dist = np.inf * np.ones((Ngtt, Nt))
            for o in range(Ngtt):
                GT = gt[GTsInFrame[0][o]][2:4]
                for e in range(Nt):
                    E = det[DetsInFrame[0][e]][2:4]
                    dist[o, e] = getDistance(GT[0], GT[1], E[0], E[1])
            tmpai = dist.copy()
            tmpai[tmpai > td] = 1e6
            if not (tmpai == 1e6).all():
                HUN_res = np.array(linear_sum_assignment(tmpai)).T
                HUN_res = HUN_res[tmpai[HUN_res[:, 0], HUN_res[:, 1]] < td]
                if len(HUN_res) > 0:
                    u, v = HUN_res[HUN_res[:, 1].argsort()].T
                    for mmm in range(len(u)):
                        M[t - 1, u[mmm]] = v[mmm] + 1

        curdetected, = np.where(M[t - 1, :])
        c[0][t - 1] = curdetected.shape[0]
        for ct in curdetected:
            eid = int(M[t - 1, ct] - 1)
            gtX = gt[GTsInFrame[0][ct], 2]
            gtY = gt[GTsInFrame[0][ct], 3]
            stX = det[DetsInFrame[0][eid], 2]
            stY = det[DetsInFrame[0][eid], 3]
            distances[t - 1, ct] = getDistance(gtX, gtY, stX, stY)
        fp[0][t - 1] = Nt - c[0][t - 1]
        m[0][t - 1] = g[0][t - 1] - c[0][t - 1]

    MODP = sum(1 - distances[distances < td] / td) / np.sum(c) * 100 \
        if sum(1 - distances[distances < td] / td) / np.sum(c) * 100 > 0 else 0
    MODA = (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 \
        if (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 > 0 else 0
    recall = np.sum(c) / np.sum(g) * 100 \
        if np.sum(c) / np.sum(g) * 100 > 0 else 0
    precision = np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 \
        if np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 > 0 else 0

    return recall, precision, MODA, MODP


def CLEAR_MOD_HUN_extended(gt, det):
    """
    Same as CLEAR_MOD_HUN but also returns per-frame matched pairs with gt IDs.

    Args:
        gt:  (N, 5) array — columns [frame, person_id, x, y, gt_id]
        det: (M, 4) array — columns [frame, det_id, x, y]

    Returns:
        recall, precision, MODA, MODP, matched_pairs (dict[frame -> list[dict]])
    """
    td = 50 / 2.5

    F = int(max(gt[:, 0])) + 1
    Ngt = int(max(gt[:, 1])) + 1

    M = np.zeros((F, Ngt))
    c = np.zeros((1, F))
    fp = np.zeros((1, F))
    m = np.zeros((1, F))
    g = np.zeros((1, F))
    distances = np.inf * np.ones((F, Ngt))
    matched_pairs = defaultdict(list)

    for t in range(1, F + 1):
        GTsInFrame = np.where(gt[:, 0] == t - 1)[0]
        DetsInFrame = np.where(det[:, 0] == t - 1)[0]
        GTsInFrame = GTsInFrame.reshape(1, -1)
        DetsInFrame = DetsInFrame.reshape(1, -1)

        Ngtt = GTsInFrame.shape[1]
        Nt = DetsInFrame.shape[1]
        g[0, t - 1] = Ngtt

        if Ngtt > 0 and Nt > 0:
            dist = np.inf * np.ones((Ngtt, Nt))
            for o in range(Ngtt):
                GT = gt[GTsInFrame[0][o]][2:4]
                for e in range(Nt):
                    E = det[DetsInFrame[0][e]][2:4]
                    dist[o, e] = getDistance(GT[0], GT[1], E[0], E[1])
            tmpai = dist.copy()
            tmpai[tmpai > td] = 1e6
            if not (tmpai == 1e6).all():
                HUN_res = np.array(linear_sum_assignment(tmpai)).T
                HUN_res = HUN_res[tmpai[HUN_res[:, 0], HUN_res[:, 1]] < td]
                if len(HUN_res) > 0:
                    u, v = HUN_res[HUN_res[:, 1].argsort()].T
                    for mmm in range(len(u)):
                        M[t - 1, u[mmm]] = v[mmm] + 1

        curdetected, = np.where(M[t - 1, :])
        c[0][t - 1] = curdetected.shape[0]
        for ct in curdetected:
            eid = int(M[t - 1, ct] - 1)
            gtX = gt[GTsInFrame[0][ct], 2]
            gtY = gt[GTsInFrame[0][ct], 3]
            gt_id = gt[GTsInFrame[0][ct], 4]
            stX = det[DetsInFrame[0][eid], 2]
            stY = det[DetsInFrame[0][eid], 3]
            distances[t - 1, ct] = getDistance(gtX, gtY, stX, stY)
            matched_pairs[t - 1].append({
                'gt_id': int(gt_id),
                'gt_xy': (gtX, gtY),
                'det_xy': (stX, stY),
                'distance': distances[t - 1, ct],
            })
        fp[0][t - 1] = Nt - c[0][t - 1]
        m[0][t - 1] = g[0][t - 1] - c[0][t - 1]

    MODP = sum(1 - distances[distances < td] / td) / np.sum(c) * 100 \
        if sum(1 - distances[distances < td] / td) / np.sum(c) * 100 > 0 else 0
    MODA = (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 \
        if (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 > 0 else 0
    recall = np.sum(c) / np.sum(g) * 100 \
        if np.sum(c) / np.sum(g) * 100 > 0 else 0
    precision = np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 \
        if np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 > 0 else 0

    return recall, precision, MODA, MODP, matched_pairs


def iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0


def compute_ap(recalls, precisions):
    """Compute AP via area-under-curve interpolation."""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])


def evaluate_map(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute mean Average Precision (mAP).

    Args:
        predictions:  dict image_id -> [[x1, y1, x2, y2, score, class_id], ...]
        ground_truths: dict image_id -> [[x1, y1, x2, y2, class_id], ...]
    """
    aps = []
    class_ids = set()
    for gt_list in ground_truths.values():
        for gt in gt_list:
            class_ids.add(gt[-1])
    for pred_list in predictions.values():
        for pred in pred_list:
            class_ids.add(pred[-1])

    for cls in class_ids:
        cls_preds = []
        cls_gts = {}
        for image_id in predictions:
            preds = [p for p in predictions[image_id] if p[5] == cls]
            cls_preds.extend([(image_id, p[:4], p[4]) for p in preds])
        for image_id in ground_truths:
            gts = [g for g in ground_truths[image_id] if g[4] == cls]
            cls_gts[image_id] = {'boxes': gts, 'matched': [False] * len(gts)}

        cls_preds.sort(key=lambda x: x[2], reverse=True)
        tp, fp = [], []
        for image_id, pred_box, score in cls_preds:
            gt_info = cls_gts.get(image_id, {'boxes': [], 'matched': []})
            max_iou, max_iou_idx = 0, -1
            for idx, gt in enumerate(gt_info['boxes']):
                iou_val = iou(pred_box, gt[:4])
                if iou_val > max_iou:
                    max_iou, max_iou_idx = iou_val, idx
            if max_iou >= iou_threshold and not gt_info['matched'][max_iou_idx]:
                tp.append(1)
                fp.append(0)
                gt_info['matched'][max_iou_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        recalls = tp_cum / (sum(len(v['boxes']) for v in cls_gts.values()) + 1e-6)
        aps.append(compute_ap(recalls, precisions))

    return np.mean(aps) if aps else 0.0


def single_keypoint_nms(kpts_db, dist_thr, max_dets=-1, combined_input=True):
    """
    Greedy NMS for single-keypoint detections.

    Args:
        kpts_db: [N, 3+] array (x, y, score, ...) when combined_input=True,
                 else list of dicts with 'keypoints' and 'score'.
        dist_thr: Euclidean distance threshold for suppression.
        max_dets: Max detections to keep (-1 = unlimited).
        combined_input: Whether input is a raw numpy array.

    Returns:
        List[int]: kept indices after NMS.
    """
    assert dist_thr > 0

    if combined_input:
        kpts = np.array(kpts_db[:, :2])
        scores = np.array(kpts_db[:, 2])
    else:
        kpts = np.array([k['keypoints'][:2] for k in kpts_db])
        scores = np.array([k['score'] for k in kpts_db])

    num_points = kpts.shape[0]
    if num_points == 0:
        return []

    diff = kpts[:, None, :] - kpts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    keep_inds = []
    suppressed = np.zeros(num_points, dtype=bool)
    for i in np.argsort(scores)[::-1]:
        if suppressed[i]:
            continue
        keep_inds.append(i)
        suppressed[dists[i] < dist_thr] = True

    if max_dets > 0:
        keep_inds = keep_inds[:max_dets]
    return keep_inds


def soft_nms_keypoints(kpts_db, dist_thr, max_dets=-1, combined_input=True,
                       method='gaussian', sigma=0.1, min_score=0.5):
    """
    Soft-NMS for single-keypoint detections with Gaussian or linear score decay.
    """
    assert dist_thr > 0
    assert method in ['gaussian', 'linear']

    if combined_input:
        kpts = np.array(kpts_db[:, :2])
        scores = np.array(kpts_db[:, 2], dtype=float)
    else:
        kpts = np.array([k['keypoints'][:2] for k in kpts_db])
        scores = np.array([k['score'] for k in kpts_db], dtype=float)

    N = len(scores)
    if N == 0:
        return []

    indices = np.arange(N)
    keep_inds = []

    while len(indices) > 0:
        current = indices[np.argmax(scores[indices])]
        keep_inds.append(current)
        dists = np.linalg.norm(kpts[indices] - kpts[current], axis=1)
        if method == 'gaussian':
            decay = np.exp(-(dists ** 2) / (2 * sigma ** 2))
        else:
            decay = np.clip(1 - dists / dist_thr, 0, 1)
        scores[indices] *= decay
        remaining = indices[scores[indices] >= min_score]
        indices = remaining[remaining != current]

    if max_dets > 0:
        keep_inds = sorted(keep_inds, key=lambda i: -scores[i])[:max_dets]
    return keep_inds


def single_keypoint_soft_nms(kpts_db, dist_thr, sigma=0.5, score_thresh=1e-3,
                              max_dets=-1, combined_input=True):
    """
    Gaussian soft-NMS for single-keypoint detections.

    Returns:
        keep_inds (List[int]), updated scores (np.ndarray [1, N])
    """
    assert dist_thr > 0

    if combined_input:
        kpts = np.array(kpts_db[:, :2])
        scores = np.array(kpts_db[:, 2], dtype=float)
    else:
        kpts = np.array([k['keypoints'][:2] for k in kpts_db])
        scores = np.array([k['score'] for k in kpts_db], dtype=float)

    num_points = kpts.shape[0]
    if num_points == 0:
        return [], np.zeros((1, 0))

    keep_inds = []
    alive = np.ones(num_points, dtype=bool)

    while True:
        valid_scores = scores.copy()
        valid_scores[~alive] = -np.inf
        i = np.argmax(valid_scores)
        if valid_scores[i] < score_thresh:
            break
        keep_inds.append(i)
        alive[i] = False
        dists = np.linalg.norm(kpts - kpts[i], axis=1)
        nearby_mask = (dists < dist_thr) & alive
        decay = np.exp(-(dists[nearby_mask] ** 2) / (2 * (dist_thr * sigma) ** 2))
        scores[nearby_mask] *= decay

    if max_dets > 0:
        keep_inds = keep_inds[:max_dets]
    return keep_inds, np.expand_dims(scores, 0)


def box_iou_numpy(boxes1, boxes2):
    """
    Compute pairwise IoU and union between two sets of boxes.

    Args:
        boxes1: [N, 4] array
        boxes2: [M, 4] array

    Returns:
        iou: [N, M], union: [N, M]
    """
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / np.clip(union, 1e-8, None), union


def keypoint_bbox_nms(kpts_db, bbox_db, dist_thr, max_dets=-1, combined_input=True):
    """
    Greedy NMS for keypoints with optional per-view bounding-box IoU gating.
    """
    assert dist_thr > 0

    if combined_input:
        kpts = np.array(kpts_db[:, :2])
        scores = np.array(kpts_db[:, 2])
    else:
        kpts = np.array([k['keypoints'][:2] for k in kpts_db])
        scores = np.array([k['score'] for k in kpts_db])

    num_points = kpts.shape[0]
    if num_points == 0:
        return []

    diff = kpts[:, None, :] - kpts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    keep_inds = []
    suppressed = np.zeros(num_points, dtype=bool)
    for i in np.argsort(scores)[::-1]:
        if suppressed[i]:
            continue
        keep_inds.append(i)
        nearby = dists[i] < dist_thr
        if np.any(nearby):
            for j in np.where(nearby)[0]:
                if j != i:
                    iou_vals, _ = box_iou_numpy(bbox_db[i], bbox_db[j])
                    if len(np.where(iou_vals != 0)[0]) == 0:
                        nearby[j] = False
        suppressed[nearby] = True

    if max_dets > 0:
        keep_inds = keep_inds[:max_dets]
    return keep_inds
