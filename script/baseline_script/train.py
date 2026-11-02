import logging
import warnings
from time import time

import numpy as np
import torch
from tqdm import tqdm

from utils import fix_seed, make_loss_graph, make_acc_graph, format_metrics
from eval import evaluate

warnings.filterwarnings("ignore", category=UserWarning)


def train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function, metric_func, gene_name_dict, DEGRank, Uscore_org):
    fix_seed(args.seed)

    ndcg_names = ["ndcgBinary@%d" % k for k in args.ndcg_topk]
    metric_names = ["mse", "pearsonr", "spearman", "DEG_spearman"] + ndcg_names
    select_metric = "val_ndcgBinary@%d" % max(args.ndcg_topk)
    log_dict = {"train_mse_loss": []}
    for split in ["val", "test"]:
        for name in metric_names:
            log_dict["%s_%s" % (split, name)] = []

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("%s/log_dict/fold=%d_seed=%d_training_setting.log" %  (args.output_path, args.fold, args.seed))
    logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])
    logging.info(args)

    # the best model is selected by the validation nDCG at the largest k
    best_val_ndcg, best_epoch = 0, 0

    for epoch in range(args.num_epochs):
        ############ train ###################
        s_time = time()
        model.train()
        losses = []
        for data in tqdm(train_loader, leave=False):
            patch, st, slide_ids = data["patch"], data["st"], data["patient_id"]
            patch, st = patch.to(args.device), st.to(args.device)

            y = model(patch)

            loss = loss_function(y["y"], st, np.array(slide_ids)).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        log_dict["train_mse_loss"].append(np.array(losses).mean())

        e_time = time()
        logging.info('[Epoch: %d/%d (%ds)] train loss: %.4f' %
                     (epoch+1, args.num_epochs, e_time-s_time, log_dict["train_mse_loss"][-1]))

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

        make_loss_graph(args, log_dict['train_mse_loss'], None, "%s/loss_graph/fold=%d_seed=%d_loss-graph.png" % (args.output_path, args.fold, args.seed))
        for name in metric_names:
            make_acc_graph(args, None, log_dict["val_" + name], log_dict["test_" + name],
                           "%s/acc_graph/fold=%d_seed=%d_%s-graph.png" % (args.output_path, args.fold, args.seed, name))

        np.save("%s/log_dict/fold=%d_seed=%d_log" % (args.output_path, args.fold, args.seed) , log_dict)
    return
