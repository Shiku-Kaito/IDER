import argparse
import os
import numpy as np
import torch
from utils import fix_seed, make_folder
from paths import DATA_ROOT, RESULT_ROOT
from get_module import get_module
from eval import eval_net

def main(args):
    fix_seed(args.seed) 
    train_net, model, optimizer, loss_function, train_loader, val_loader, test_loader, metric_func, gene_name_dict, DEGRank, Uscore = get_module(args)
 
    args.output_path += '%s/eval_top_genes=%d_TrainCluster_k=%d/w_pretrain_lr=3e-4/%s_loss=%s/' % (args.dataset, len(gene_name_dict["eval"]), args.clustering_k, args.mode, args.loss) 
    make_folder(args)
   
    if args.is_evaluation == False:
        train_net(args, model, optimizer, train_loader, val_loader, test_loader, loss_function, metric_func, gene_name_dict, DEGRank, Uscore)
        return

    model.load_state_dict(torch.load(("%s/model/fold=%d_seed=%d-best_model_cluster.pkl") % (args.output_path, args.fold, args.seed), map_location=args.device))
    return eval_net(args, model, test_loader, gene_name_dict, DEGRank, Uscore)


if __name__ == '__main__':
    results_dict = {name: [] for name in METRIC_NAMES}

    fold_parser = argparse.ArgumentParser(add_help=False)
    fold_parser.add_argument('--folds', default=[0, 1, 2, 3], type=int, nargs='+', help='folds to run (default: all 4)')
    fold_parser.add_argument('--ndcg_topk', default=[50, 100, 200], type=int, nargs='+', help='k values of nDCG@k (the largest k selects the best model)')
    pre_args = fold_parser.parse_known_args()[0]
    folds = pre_args.folds
    METRIC_NAMES = ["test_DEG_spearman"] + ["ndcgBinary@%d" % k for k in pre_args.ndcg_topk]

    for fold in folds:
        parser = argparse.ArgumentParser()
        parser.add_argument('--folds', default=folds, type=int, nargs='+', help='folds to run (default: all 4)')
        parser.add_argument('--ndcg_topk', default=pre_args.ndcg_topk, type=int, nargs='+', help='k values of nDCG@k (the largest k selects the best model)')
        # Data selection
        parser.add_argument('--fold', default=fold,
                            type=int, help='fold number')
        parser.add_argument('--dataset', default='Her2st',
                            type=str, help='')
        parser.add_argument('--labeled_slides', default=["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"],
                            type=list, help='')
        # Training setup
        parser.add_argument('--num_epochs', default=1000, type=int,
                            help='number of epochs for training.')
        parser.add_argument('--device', default='cuda:0',
                            type=str, help='device')
        parser.add_argument('--batch_size', default=1024,
                            type=int, help='batch size for training.')
        parser.add_argument('--seed', default=0,
                            type=int, help='seed value')
        parser.add_argument('--num_workers', default=0, type=int,
                            help='number of workers for training.')
        parser.add_argument('--lr', default=3e-4,
                            type=float, help='learning rate')
        parser.add_argument('--is_evaluation', default=1,
                            type=int, help='0: train, 1: evaluate')
        # Loss / module selection
        parser.add_argument('--loss', default='pcc',
                            type=str, help="mse or pcc (the loss of the baseline module; IDER uses pcc)")
        parser.add_argument('--module', default='IDER',
                            type=str, help="IDER or baseline")
        parser.add_argument('--mode', default='',    # set in get_module
                            type=str, help="")
        # Paths
        parser.add_argument('--output_path',
                            default=RESULT_ROOT + '/', type=str, help="output root directory")
        parser.add_argument('--input_path',
                            default=os.path.dirname(DATA_ROOT), type=str, help="directory that contains data/ (input root)")
        # Clustering
        parser.add_argument('--clustering_k',
                            default=10, type=int, help="number of clusters (k-means)")
        parser.add_argument('--random_state',
                            default=5, type=int, help="")
        # IDER (DEG-rank loss + PCC loss)
        parser.add_argument('--PCCLoss_lambda',
                            default=1.0, type=float, help="weight of the PCC loss")
        parser.add_argument('--temperature',
                            default=0.01, type=float, help="temperature of the differentiable U-score")
        args = parser.parse_args()

        if args.is_evaluation == False:
            main(args)
        else:
            result_dict = main(args)
            for name in METRIC_NAMES:
                results_dict[name].append(result_dict[name])
            print("############ fold=%d ###########" % args.fold)
            print("@ " + ", ".join("%s:%.5f" % (name, float(result_dict[name])) for name in METRIC_NAMES))

    if args.is_evaluation == True:
        lines = ["############ %d-fold cross-validation ###########" % len(folds)]
        lines.append("@ " + ", ".join(
            "%s:%.3f$\\pm$%.3f" % (name, np.mean(results_dict[name]), np.std(results_dict[name])) for name in METRIC_NAMES))
        print("\n".join(lines))

        save_path = "%s/test_metrics/result.txt" % args.output_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("\n".join(lines) + "\n")
