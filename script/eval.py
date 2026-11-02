from collections import Counter
import copy
import numpy as np
import torch
from tqdm import tqdm
from culc_DEG_metrics import DEG_ranking, DEG_metric_func


def eval_net(args, model, test_loader, gene_name_dict, DEGRank, Uscore_org):
    """Evaluate DEG-ranking performance on the test slides.

    Returns the slide-averaged DEG Spearman and nDCG (binary) @k for k in args.ndcg_topk.
    """
    model.eval()
    pred_st, slide_ids_list, cluster_ids_list = [], [], []
    with torch.no_grad():
        for data in tqdm(test_loader, leave=False):
            patch, slide_ids = data["patch"].to(args.device), data["patient_id"]

            y = model(patch)

            pred_st.extend(y["y"].cpu().detach().numpy())
            slide_ids_list.extend(slide_ids)
            cluster_ids_list.extend(data["cluster_id"])

    pred_st, slide_ids_list = np.array(pred_st), np.array(slide_ids_list)
    cluster_ids_list = np.array(cluster_ids_list)

    ##### Remove minor clusters and apply patient-wise filtering #####
    # clusters that are always excluded
    always_remove = {6, 1, 5}
    minor_clusters_per_patient = {}
    for slide in np.unique(slide_ids_list):
        counter = Counter(cluster_ids_list[slide_ids_list == slide])
        # clusters with fewer than 10 spots, plus the fixed exclusions
        minor_clusters = {c for c, cnt in counter.items() if cnt < 10}
        minor_clusters_per_patient[slide] = list(minor_clusters | always_remove)
    print(minor_clusters_per_patient)

    Uscore = copy.deepcopy(Uscore_org)

    ndcg_names = ["ndcgBinary@%d" % k for k in args.ndcg_topk]
    DEG_sum = {name: 0 for name in ["spearman"] + ndcg_names}
    for p in np.unique(slide_ids_list):
        undetermin_mask = np.array(cluster_ids_list)[slide_ids_list==p]!=6
        DEG_rank_pred, Uscore_pred =  DEG_ranking(X = torch.tensor(pred_st)[slide_ids_list==p][undetermin_mask], cluster_id = np.array(cluster_ids_list)[slide_ids_list==p][undetermin_mask], gene_chunk = 50, device=args.device)

        remove_clusters = set(minor_clusters_per_patient[p])

        Uscore_pred = {k: v for k, v in Uscore_pred.items() if k not in remove_clusters}
        Uscore["test"][p] = {k: v for k, v in Uscore["test"][p].items() if k not in remove_clusters}

        DEG_rank_pred = {k: v for k, v in DEG_rank_pred.items() if k not in remove_clusters}
        DEGRank["test"][p] = {k: v for k, v in DEGRank["test"][p].items() if k not in remove_clusters}

        test_DEG_metric = DEG_metric_func(pred_Uscore=Uscore_pred, gt_Uscore=Uscore["test"][p], topk_list=args.ndcg_topk)
        for name in DEG_sum:
            DEG_sum[name] += test_DEG_metric[name]

    n_slides = len(np.unique(slide_ids_list))
    result = {"test_DEG_spearman": DEG_sum["spearman"] / n_slides}
    for name in ndcg_names:
        result[name] = DEG_sum[name] / n_slides
    return result


def evaluate(args, model, loader, metric_func, gene_name_dict, Uscore_split):
    """Evaluate the model on val / test: expression metrics and slide-averaged DEG metrics.

    Uscore_split: ground-truth U-scores of the split, {patient_id: {cluster id: U-score}}
    """
    model.eval()
    gt_st, pred_st, slide_ids_list, cluster_ids_list = [], [], [], []
    with torch.no_grad():
        for data in tqdm(loader, leave=False):
            patch, st = data["patch"].to(args.device), data["st"].to(args.device)

            y = model(patch)

            gt_st.extend(st.cpu().detach().numpy()), pred_st.extend(y["y"].cpu().detach().numpy())
            slide_ids_list.extend(data["patient_id"])
            cluster_ids_list.extend(data["cluster_id"].tolist())

    gt_st, pred_st, slide_ids_list = np.array(gt_st), np.array(pred_st), np.array(slide_ids_list)
    cluster_ids_list = np.array(cluster_ids_list)
    result = dict(metric_func(gt_st, pred_st, slide_ids_list, gene_name_dict))

    # slide-averaged DEG metrics: {name in the DEG metric dict: name in the result}
    DEG_names = {"spearman": "DEG_spearman", **{"ndcgBinary@%d" % k: "ndcgBinary@%d" % k for k in args.ndcg_topk}}
    DEG_sum = {name: 0 for name in DEG_names}
    unique_slide_ids = np.unique(slide_ids_list)
    for p in unique_slide_ids:
        _, Uscore_pred = DEG_ranking(X=torch.tensor(pred_st)[slide_ids_list==p], cluster_id=cluster_ids_list[slide_ids_list==p], gene_chunk=50, device=args.device)
        DEG_metric = DEG_metric_func(pred_Uscore=Uscore_pred, gt_Uscore=Uscore_split[p], topk_list=args.ndcg_topk)
        for name in DEG_names:
            DEG_sum[name] += DEG_metric[name]
    for name, result_name in DEG_names.items():
        result[result_name] = DEG_sum[name] / len(unique_slide_ids)
    return result
