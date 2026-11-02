import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchmetrics.regression import PearsonCorrCoef


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the deterministic algorithm


def make_folder(args):
    """Create the output directories under args.output_path."""
    for sub_dir in ["", "acc_graph", "log_dict", "loss_graph", "model", "test_metrics"]:
        os.makedirs(os.path.join(args.output_path, sub_dir), exist_ok=True)


def culc_metric(gt_st, pred_st, slide_ids, gene_name_dict):
    """Slide-averaged MSE and Pearson correlation (mean over genes) between the ground truth and the prediction.

    "spearman" is not computed (always 0); the key is kept for the training logs.
    """
    mse_all, pearsonr_all = 0, 0
    unique_slide_ids = np.unique(slide_ids)
    for slide_id in unique_slide_ids:
        mse = ((gt_st[slide_ids==slide_id]-pred_st[slide_ids==slide_id])**2).mean()

        metric_pearson = PearsonCorrCoef(num_outputs=gt_st.shape[-1])
        pearsonr = metric_pearson(torch.tensor(pred_st[slide_ids==slide_id]), torch.tensor(gt_st[slide_ids==slide_id]))
        pearsonr = pearsonr.nanmean()

        mse_all += mse
        pearsonr_all += pearsonr

    mse_all /= len(unique_slide_ids)
    pearsonr_all /= len(unique_slide_ids)
    return {"mse": mse_all, "pearsonr": pearsonr_all, "spearman": 0}


def make_loss_graph(args, keep_train_loss, keep_valid_loss, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(keep_train_loss, label = 'train')
    if keep_valid_loss is not None:
        ax.plot(keep_valid_loss, label = 'valid')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("Losses")
    plt.legend()
    fig.savefig(path)
    plt.close()


def make_acc_graph(args, train_bag_acc, val_bag_acc, test_bag_acc, path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if train_bag_acc is not None:
        ax.plot(train_bag_acc, label = 'train')
    ax.plot(val_bag_acc, label = 'valid')
    ax.plot(test_bag_acc, label = 'test')
    ax.set_xlabel("Epoch numbers")
    ax.set_ylabel("accuracy")
    plt.legend()
    fig.savefig(path)
    plt.close()


def format_metrics(log_dict, split, metric_names, idx):
    """One-line summary of the logged metrics of `split` at index `idx`."""
    return ", ".join("%s: %.4f" % (name, log_dict["%s_%s" % (split, name)][idx]) for name in metric_names)
