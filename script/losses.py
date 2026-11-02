import torch

_eps = 1e-6

# ===== Baseline loss (IDER) =====
def PearsonCorrCoef_loss(x, y, slide_ids):
    mean_x = torch.mean(x, dim=0, keepdim=True)  # (1, 512)
    mean_y = torch.mean(y, dim=0, keepdim=True)  # (1, 512)

    xm = x - mean_x  
    ym = y - mean_y 

    r_num = torch.sum(xm * ym, dim=0)  # (512,)
    r_den = torch.sqrt(torch.sum(xm ** 2, dim=0)) * torch.sqrt(torch.sum(ym ** 2, dim=0))  # (512,)

    pearson_corr = r_num / (r_den + 1e-8)  # avoid division by zero
    pearson_corr = torch.nan_to_num(pearson_corr, nan=-1)  # replace NaN with -1
    return 1-pearson_corr


def MSE_loss(x, y, slide_ids):
    loss = ((x-y)**2)        
    return torch.mean(loss)


# ===== DEG-rank loss (IDER) =====
def Diff_Uscore(
    y_pred: torch.Tensor,        # [B, G]  (larger = higher)
    cluster_ids: torch.Tensor,   # [B]
    sampling_cid: torch.Tensor,
    temperature: float = 1.0,    # T
    eps: float = 1e-8,
):
    """
    soft Wilcoxon U (soft version of the number of times A beats B):
      U[g] = sum_{a in A} sum_{b in B} sigmoid((A[a,g] - B[b,g]) / T)

    Returns:
      deg_scores: dict[cid -> Tensor[G]]
    """
    deg_scores = {}
    # temperature, eps = 0.01, 1e-8
    T = float(temperature) + float(eps)
    for cid in sampling_cid.tolist():
        mask_a = (cluster_ids == cid)
        mask_b = ~mask_a

        na = int(mask_a.sum().item())
        nb = int(mask_b.sum().item())
        if na == 0 or nb == 0:
            continue
        
        A = y_pred[mask_a]
        B = y_pred[mask_b]

        U = torch.sigmoid((A[:, None, :] - B[None, :, :]) / T)  # [na, nb, G]
        U = U.sum(dim=(0, 1))  # [G]

        
        deg_scores[cid] = U

    return deg_scores



def DEGrank_loss(
    Uscore_pred,
    Uscore_gt,
    cid,
):
    """
    DEG-rank loss of one cluster: 1 - PCC between the predicted and ground-truth U-scores of cluster `cid`,
    each normalized by its sum.
    (In training, `cid` is one of 2 randomly sampled clusters: the last one of the sampled pair.)

    Uscore_pred, Uscore_gt: dict {cluster id: U-score [G]}
    """
    pred = Uscore_pred[cid] / sum(Uscore_pred[cid])
    gt = Uscore_gt[cid] / sum(Uscore_gt[cid])

    xm = pred - torch.mean(pred, dim=0, keepdim=True)
    ym = gt - torch.mean(gt, dim=0, keepdim=True)

    r_num = torch.sum(xm * ym, dim=0)
    r_den = torch.sqrt(torch.sum(xm ** 2, dim=0)) * torch.sqrt(torch.sum(ym ** 2, dim=0))

    pearson_corr = r_num / (r_den + 1e-8)  # avoid division by zero
    pearson_corr = torch.nan_to_num(pearson_corr, nan=-1)  # replace NaN with -1
    return 1 - pearson_corr
