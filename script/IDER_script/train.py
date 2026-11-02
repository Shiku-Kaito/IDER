import copy
import logging
import warnings
from time import time

import numpy as np
import torch
from tqdm import tqdm

from utils import fix_seed, make_loss_graph, make_acc_graph, format_metrics
from losses import Diff_Uscore
from eval import evaluate

warnings.filterwarnings("ignore", category=UserWarning)


def train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function, metric_func, gene_name_dict, DEGRank, Uscore_org):
    fix_seed(args.seed)

    ndcg_names = ["ndcgBinary@%d" % k for k in args.ndcg_topk]
    metric_names = ["mse", "pearsonr", "spearman", "DEG_spearman"] + ndcg_names
    select_metric = "val_ndcgBinary@%d" % max(args.ndcg_topk)
    log_dict = {"train_loss": [], "train_DEG_loss": [], "train_pcc_loss": []}
    for split in ["val", "test"]:
        for name in metric_names:
            log_dict["%s_%s" % (split, name)] = []

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("%s/log_dict/fold=%d_seed=%d_training_setting.log" %  (args.output_path, args.fold, args.seed))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.info(args)

    DEGrank_loss_func, PCC_loss_func = loss_function["DEGrank"], loss_function["pcc"]

    # ground-truth U-scores of the training patients (on the device)
    Uscore_train = copy.deepcopy(Uscore_org['train'])
    for p in Uscore_train:
        for cid in Uscore_train[p]:
            Uscore_train[p][cid] = Uscore_train[p][cid].to(args.device)

    # the best model is selected by the validation nDCG at the largest k
    best_val_ndcg, best_epoch = 0, 0

    for epoch in range(args.num_epochs):
        ############ train ###################
        s_time = time()
        model.train()
        losses, losses_DEG, losses_PCC = [], [], []
        for data in tqdm(train_loader, leave=False):
            patch, st, slide_ids, cluster_ids = data["patch"], data["st"], data["patient_id"], data["cluster_id"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = model(patch)

            # DEG-rank loss: 2 clusters are sampled at random and the last one is used for the loss
            cluster_ids = torch.as_tensor(cluster_ids)
            unique_cids = torch.unique(cluster_ids)
            sampling_cid = unique_cids[torch.randperm(len(unique_cids))[:2]][-1:]
            Uscore_pred = Diff_Uscore(
                            y_pred = y["y"],
                            cluster_ids = cluster_ids,
                            sampling_cid = sampling_cid,
                            temperature = args.temperature
                        )
            loss_DEG = DEGrank_loss_func(Uscore_pred, Uscore_train[slide_ids[0]], sampling_cid.item())
            loss_PCC = PCC_loss_func(y["y"], st, np.array(slide_ids)).mean()
            loss = loss_DEG + args.PCCLoss_lambda * loss_PCC

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item()), losses_DEG.append(loss_DEG.item()), losses_PCC.append(loss_PCC.item())

        log_dict["train_loss"].append(np.array(losses).mean()), log_dict["train_DEG_loss"].append(np.array(losses_DEG).mean()), log_dict["train_pcc_loss"].append(np.array(losses_PCC).mean())

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] train loss: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["train_loss"][-1]))

        ################# validation / test ####################
        for split, loader in [("val", val_loader), ("test", test_loader)]:
            s_time = time()
            result = evaluate(args, model, loader, metric_func, gene_name_dict, Uscore_org[split])
            for name in metric_names:
                log_dict["%s_%s" % (split, name)].append(result[name])

            e_time = time()
            logging.info('[Epoch: %d/%d (%ds)], @ %s' % (epoch+1, args.num_epochs, e_time-s_time, format_metrics(log_dict, split, metric_names, -1)))

        if best_val_ndcg < log_dict[select_metric][-1]:
            best_val_ndcg = log_dict[select_metric][-1]
            best_epoch = epoch
            torch.save(model.state_dict(), ("%s/model/fold=%d_seed=%d-best_model_cluster.pkl") % (args.output_path, args.fold, args.seed))

        logging.info("best epoch: %d , %s" % (best_epoch + 1, format_metrics(log_dict, "test", metric_names, best_epoch)))

        for loss_name, graph_name in [("train_loss", "loss"), ("train_DEG_loss", "loss_DEG"), ("train_pcc_loss", "loss_PCC")]:
            make_loss_graph(args, log_dict[loss_name], None, "%s/loss_graph/fold=%d_seed=%d_%s-graph.png" % (args.output_path, args.fold, args.seed, graph_name))
        for name in metric_names:
            make_acc_graph(args, None, log_dict["val_" + name], log_dict["test_" + name],
                           "%s/acc_graph/fold=%d_seed=%d_%s-graph.png" % (args.output_path, args.fold, args.seed, name))

        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
    return
