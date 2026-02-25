from util.data import *
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, precision_recall_curve, auc


def _to_numpy_array(data):
    return np.array(data, dtype=np.float64)


def _compute_residual_scores(pred, gt):
    pred_np = _to_numpy_array(pred)
    gt_np = _to_numpy_array(gt)
    delta = np.abs(pred_np - gt_np)

    if delta.ndim == 1:
        return delta

    reduce_axes = tuple(range(1, delta.ndim))
    return np.mean(delta, axis=reduce_axes)


def compute_condition_gate(test_pi):
    pi_np = _to_numpy_array(test_pi)
    if pi_np.ndim != 2 or pi_np.shape[0] == 0:
        return np.ones((0,), dtype=np.float64)

    delta_pi = np.zeros((pi_np.shape[0],), dtype=np.float64)
    delta_pi[1:] = np.sum(np.abs(pi_np[1:] - pi_np[:-1]), axis=1)
    gate = 1.0 - 0.5 * delta_pi

    return gate


def compute_structural_drift_scores(test_pi, model, topq=0.1, chunk_size=256):
    pi_np = _to_numpy_array(test_pi)
    if pi_np.ndim != 2 or pi_np.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    device = next(model.parameters()).device

    with torch.no_grad():
        e_base = model.e_base.detach().to(device)
        low_rank_u = model.low_rank_u.detach().to(device)
        low_rank_v = model.low_rank_v.detach().to(device)

        low_rank_delta = torch.matmul(low_rank_u, low_rank_v)
        proto_embed = e_base.unsqueeze(0) + low_rank_delta

        total_steps = pi_np.shape[0]
        num_nodes = int(proto_embed.shape[1])
        topq_k = max(1, int(np.ceil(num_nodes * float(topq))))

        struct_scores = []
        last_A = None

        for start_idx in range(0, total_steps, int(chunk_size)):
            end_idx = min(start_idx + int(chunk_size), total_steps)

            pi_chunk = torch.tensor(pi_np[start_idx:end_idx], dtype=torch.float32, device=device)
            mixed_embed = torch.einsum('tm,mnd->tnd', pi_chunk, proto_embed)
            mixed_embed = torch.nan_to_num(mixed_embed, nan=0.0, posinf=1e4, neginf=-1e4)

            A_chunk = torch.bmm(mixed_embed, mixed_embed.transpose(1, 2))
            A_chunk = torch.nan_to_num(A_chunk, nan=0.0, posinf=1e4, neginf=-1e4)

            for step_idx in range(A_chunk.shape[0]):
                A_t = A_chunk[step_idx]

                if last_A is None:
                    struct_scores.append(0.0)
                    last_A = A_t
                    continue

                row_l1_diff = torch.sum(torch.abs(A_t - last_A), dim=1)
                row_l1_prev = torch.sum(torch.abs(last_A), dim=1) + 1e-8
                drift_per_node = row_l1_diff / row_l1_prev

                top_vals, _ = torch.topk(drift_per_node, k=topq_k, dim=0)
                struct_scores.append(float(torch.mean(top_vals).detach().cpu().item()))
                last_A = A_t

    return np.array(struct_scores, dtype=np.float64)


def _compute_final_scores(result, model, alpha=1.0, gamma=1.0, topq=0.1, chunk_size=256):
    pred = result[0]
    gt = result[1]
    pi = result[3] if len(result) > 3 else []

    residual_scores = _compute_residual_scores(pred, gt)

    if len(pi) > 0:
        gate_scores = compute_condition_gate(pi)
        struct_scores = compute_structural_drift_scores(pi, model, topq=topq, chunk_size=chunk_size)
    else:
        gate_scores = np.ones_like(residual_scores, dtype=np.float64)
        struct_scores = np.zeros_like(residual_scores, dtype=np.float64)

    time_len = min(len(residual_scores), len(gate_scores), len(struct_scores))
    residual_scores = residual_scores[:time_len]
    gate_scores = gate_scores[:time_len]
    struct_scores = struct_scores[:time_len]

    final_scores = alpha * residual_scores + gamma * gate_scores * struct_scores

    return final_scores, residual_scores, struct_scores, gate_scores


def get_full_err_scores(test_result, val_result, model, alpha=1.0, gamma=1.0, topq=0.1, chunk_size=256):
    final_scores, residual_scores, struct_scores, gate_scores = _compute_final_scores(
        test_result,
        model,
        alpha=alpha,
        gamma=gamma,
        topq=topq,
        chunk_size=chunk_size,
    )

    return final_scores, residual_scores, struct_scores, gate_scores


def get_final_err_scores(test_result, val_result, model, alpha=1.0, gamma=1.0, topq=0.1, chunk_size=256):
    full_scores, _, _, _ = get_full_err_scores(
        test_result,
        val_result,
        model,
        alpha=alpha,
        gamma=gamma,
        topq=topq,
        chunk_size=chunk_size,
    )

    return full_scores


def compute_detection_delay(pred_labels, true_labels):
    pred = np.array(pred_labels).astype(int)
    true = np.array(true_labels).astype(int)

    if pred.shape[0] != true.shape[0]:
        time_len = min(pred.shape[0], true.shape[0])
        pred = pred[:time_len]
        true = true[:time_len]

    delays = []
    idx = 0
    total_len = len(true)

    while idx < total_len:
        if true[idx] == 1:
            start = idx
            while idx + 1 < total_len and true[idx + 1] == 1:
                idx += 1
            end = idx

            hit_indices = np.where(pred[start:end + 1] == 1)[0]
            if hit_indices.size > 0:
                first_hit = start + int(hit_indices[0])
                delays.append(first_hit - start)
        idx += 1

    if len(delays) == 0:
        return np.nan

    return float(np.mean(delays))


def compute_modern_metrics(scores, true_labels, threshold_steps=400):
    score_arr = _to_numpy_array(scores).reshape(-1)
    true_arr = np.array(true_labels).astype(int).reshape(-1)

    time_len = min(score_arr.shape[0], true_arr.shape[0])
    score_arr = score_arr[:time_len]
    true_arr = true_arr[:time_len]

    prec_curve, rec_curve, _ = precision_recall_curve(true_arr, score_arr)
    pr_auc = auc(rec_curve, prec_curve)

    quantiles = np.linspace(0.0, 1.0, int(threshold_steps), endpoint=False)
    candidate_thresholds = np.quantile(score_arr, quantiles)

    best_f1 = -1.0
    best_pre = 0.0
    best_rec = 0.0
    best_th = float(candidate_thresholds[0]) if len(candidate_thresholds) > 0 else 0.0
    best_pred = np.zeros_like(true_arr)

    for th in candidate_thresholds:
        pred_labels = (score_arr > th).astype(int)

        pre = precision_score(true_arr, pred_labels, zero_division=0)
        rec = recall_score(true_arr, pred_labels, zero_division=0)
        f1 = f1_score(true_arr, pred_labels, zero_division=0)

        if f1 > best_f1:
            best_f1 = float(f1)
            best_pre = float(pre)
            best_rec = float(rec)
            best_th = float(th)
            best_pred = pred_labels

    ttd = compute_detection_delay(best_pred, true_arr)

    return {
        'strict_f1': best_f1,
        'strict_precision': best_pre,
        'strict_recall': best_rec,
        'strict_pr_auc': float(pr_auc),
        'best_threshold': best_th,
        'avg_ttd': ttd,
    }


def get_err_scores(test_res, val_res):
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64),
                        np.array(test_gt).astype(np.float64)
                    ))
    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])

    return smoothed_err_scores



def get_loss(predict, gt):
    return eval_mseloss(predict, gt)


def get_f1_scores(total_err_scores, gt_labels, topk=1):
    total_err_scores = _to_numpy_array(total_err_scores)
    if total_err_scores.ndim == 1:
        final_topk_fmeas = eval_scores(total_err_scores.tolist(), gt_labels, 400)
        return final_topk_fmeas

    total_features = total_err_scores.shape[0]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    for i, indexs in enumerate(topk_indices):
        sum_score = sum(score for score in sorted([total_err_scores[index, i] for index in indexs]))
        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas


def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_err_scores = _to_numpy_array(total_err_scores)
    normal_scores = _to_numpy_array(normal_scores)

    if total_err_scores.ndim == 1:
        total_topk_err_scores = total_err_scores
    else:
        total_features = total_err_scores.shape[0]
        topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
        total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    gt_labels = np.array(gt_labels).astype(int)
    pred_labels = pred_labels.astype(int)

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, thresold


def get_best_performance_data(total_err_scores, gt_labels, topk=1):
    total_err_scores = _to_numpy_array(total_err_scores)

    if total_err_scores.ndim == 1:
        total_topk_err_scores = total_err_scores
    else:
        total_features = total_err_scores.shape[0]
        topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
        total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores.tolist(), gt_labels, 400, return_thresold=True)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    gt_labels = np.array(gt_labels).astype(int)
    pred_labels = pred_labels.astype(int)

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return max(final_topk_fmeas), pre, rec, auc_score, thresold
