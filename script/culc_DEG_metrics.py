import numpy as np
import torch
from sklearn.metrics import ndcg_score
from torchmetrics.regression import SpearmanCorrCoef
from typing import Dict, Tuple


def _ndcg_binary_at_k(pred_score, gt_score, k: int):
    """
    nDCG@k with binary relevance: the top-k genes of the ground-truth score are relevant (1), the others 0.
    pred_score, gt_score: [G] (larger = higher rank)
    """
    pred_score_1d = np.asarray(pred_score, dtype=np.float64)
    gt_score_1d   = np.asarray(gt_score,   dtype=np.float64)

    gt_rank = np.argsort(np.argsort(-gt_score_1d)) + 1

    y_true_binary = (gt_rank <= k).astype(np.float32).reshape(1, -1)
    y_score = pred_score_1d.reshape(1, -1)
    return ndcg_score(y_true_binary, y_score, k=k)


def DEG_metric_func(pred_Uscore, gt_Uscore, topk_list=(50, 100, 200)):
    """
    Compare predicted and ground-truth DEG U-scores of one slide.
    pred_Uscore, gt_Uscore: dict {cluster id: U-score [G]}
    Returns the mean over clusters of the Spearman correlation and of nDCG@k for each k in topk_list.
    """
    metric_dict = {}
    n_clusters = len(pred_Uscore.keys())

    spearman_all = 0
    for g in list(pred_Uscore.keys()):
        metric_spearman = SpearmanCorrCoef(num_outputs=1)
        spearman = metric_spearman(
            torch.as_tensor(pred_Uscore[g], dtype=torch.float32),
            torch.as_tensor(gt_Uscore[g], dtype=torch.float32),
        )
        spearman_all += spearman
    metric_dict["spearman"] = spearman_all / n_clusters

    for top_k in topk_list:
        ndcg_binary_all = 0.0
        for g in list(pred_Uscore.keys()):
            pred_score = torch.as_tensor(pred_Uscore[g], dtype=torch.float32)
            gt_score = torch.as_tensor(gt_Uscore[g], dtype=torch.float32)
            ndcg_binary_all += _ndcg_binary_at_k(pred_score, gt_score, k=top_k)
        metric_dict["ndcgBinary@%d" % top_k] = ndcg_binary_all / n_clusters

    return metric_dict


@torch.no_grad()
def DEG_ranking(
    X: torch.Tensor,                 # [N, G]
    cluster_id,                      # array-like length N (list/np/torch)
    gene_chunk: int = 50,
    device: str = "cuda",
    eps: float = 1e-12,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Mann-Whitney U statistic (one-sided "greater": cluster vs. the other clusters) of every gene, computed on the
    GPU with the rank-sum method (ties get the average rank) in gene chunks.

    Returns:
      rank_dict[cid] = gene indices sorted by descending U [G] (CPU, long)
      U_dict[cid]    = U statistic [G] (CPU, float64)
    """
    assert X.dim() == 2, "X must be [N, G]"
    N, G = X.shape

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # use float64 to reproduce SciPy's results
    if X.dtype not in (torch.float32, torch.float64):
        X = X.float()
    X = X.to(device, non_blocking=True).to(torch.float64)

    cid_t = torch.as_tensor(cluster_id, device=device)
    unique_cids = torch.unique(cid_t.to(torch.int64))

    rank_dict: Dict[int, torch.Tensor] = {}
    U_dict: Dict[int, torch.Tensor] = {}

    for cid in unique_cids.tolist():
        mask_a = (cid_t == cid)
        mask_b = ~mask_a

        na = int(mask_a.sum().item())
        nb = int(mask_b.sum().item())
        if na == 0 or nb == 0:
            continue

        A = X[mask_a]  # [na, G]
        B = X[mask_b]  # [nb, G]
        n = na + nb

        U_all = torch.empty((G,), device=device, dtype=torch.float64)

        # ranks 1..n (matching the sorted positions)
        ranks_linear = torch.arange(1, n + 1, device=device, dtype=torch.float64).view(n, 1)  # [n,1]

        for g0 in range(0, G, gene_chunk):
            g1 = min(g0 + gene_chunk, G)
            gc = g1 - g0

            A_blk = A[:, g0:g1]  # [na, gc]
            B_blk = B[:, g0:g1]  # [nb, gc]

            V = torch.cat([A_blk, B_blk], dim=0)          # [n, gc]
            sorted_vals, idx = torch.sort(V, dim=0)       # sorted_vals: [n,gc], idx: [n,gc]

            # ---------- average rank of ties (mid-rank), fully vectorized ----------
            # change[t,j]=True: a new value starts here (start of a tie block)
            change = torch.ones_like(sorted_vals, dtype=torch.bool)
            change[1:] = sorted_vals[1:] != sorted_vals[:-1]

            # group_id: id of each tie block (independent per column)
            group_id = torch.cumsum(change, dim=0) - 1  # [n, gc], 0..(#groups-1)

            # expand rank positions (1..n) to gc columns
            rl = ranks_linear.expand(n, gc)  # [n, gc]

            # flatten group_id to 1D with a per-column offset for scatter_add
            # e.g. col0: +0*n, col1: +1*n, ... to avoid collisions between columns
            offsets = (torch.arange(gc, device=device, dtype=group_id.dtype) * n).view(1, gc)  # [1,gc]
            gid_flat = (group_id + offsets).reshape(-1)  # [n*gc]
            rl_flat = rl.reshape(-1)                     # [n*gc]

            total_bins = gc * n  # at most n groups per column, so n bins per column are enough
            sum_buf = torch.zeros((total_bins,), device=device, dtype=torch.float64)
            cnt_buf = torch.zeros((total_bins,), device=device, dtype=torch.float64)

            sum_buf.scatter_add_(0, gid_flat, rl_flat)
            cnt_buf.scatter_add_(0, gid_flat, torch.ones_like(rl_flat))

            mean_buf = sum_buf / (cnt_buf + eps)  # [total_bins]

            # assign the average rank to each position (in sorted order)
            ranks_sorted = mean_buf[gid_flat].reshape(n, gc)  # [n,gc]

            # restore the original row order (unsort)
            ranks = torch.empty_like(V, dtype=torch.float64)  # [n,gc]
            ranks.scatter_(0, idx, ranks_sorted)

            # A is the first na rows of the concatenation
            R_A = ranks[:na].sum(dim=0)  # [gc]

            # U1 (number of times A beats B) for the one-sided "greater" test
            U1 = R_A - (na * (na + 1) / 2.0)  # [gc]
            U_all[g0:g1] = U1


        # rank genes by descending U (strongest in A first)
        U_rank = torch.argsort(U_all, descending=True)

        rank_dict[int(cid)] = U_rank.detach().cpu()
        U_dict[int(cid)] = U_all.detach().cpu()

    return rank_dict, U_dict
