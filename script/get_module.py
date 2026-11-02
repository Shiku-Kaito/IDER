import torch
from utils import culc_metric
from dataloader import load_dataset

from IDER_script.train import train_net as IDER_train_net
from model import ST_net

from baseline_script.train import train_net as baseline_train_net

from losses import PearsonCorrCoef_loss, MSE_loss, DEGrank_loss


def get_module(args):
    if args.module ==  "IDER":
        args.mode = "IDER_PCCLoss_lambda=%.2f" % (args.PCCLoss_lambda)
        train_loader, val_loader, test_loader, gene_name_dict, all_gene_num, DEGRank, Uscore = load_dataset(args) 

        # Model
        model = ST_net(args, all_gene_num)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        loss_function = {"pcc": PearsonCorrCoef_loss, "DEGrank": DEGrank_loss}
        metric_func = culc_metric
        # Train net
        train_net = IDER_train_net

    elif args.module ==  "baseline":
        # baseline: ST-Net trained with a single loss (--loss mse or pcc)
        args.mode = "ST_net"
        train_loader, val_loader, test_loader, gene_name_dict, all_gene_num, DEGRank, Uscore = load_dataset(args) 

        # Model
        model = ST_net(args, all_gene_num)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        if args.loss == 'mse':
            loss_function = MSE_loss
        elif args.loss == 'pcc':
            loss_function = PearsonCorrCoef_loss
        else:
            raise ValueError("baseline supports --loss mse or pcc")
            
        metric_func = culc_metric
        # Train net
        train_net = baseline_train_net

    else:
        raise ValueError("Unknown module: %s" % args.module)

    return train_net, model, optimizer, loss_function, train_loader, val_loader, test_loader, metric_func, gene_name_dict, DEGRank, Uscore
