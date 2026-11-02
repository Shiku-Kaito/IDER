# Proposed method (IDER): DEG-rank loss + PCC loss. is_evaluation 0: train, 1: evaluate
python main.py --module "IDER" --batch_size 1024 --loss "pcc" --is_evaluation 0
python main.py --module "IDER" --batch_size 1024 --loss "pcc" --is_evaluation 1

# Baselines (ST-Net with a single loss): mse / pcc
python main.py --module "baseline" --batch_size 1024 --loss "mse" --is_evaluation 0
python main.py --module "baseline" --batch_size 1024 --loss "mse" --is_evaluation 1
python main.py --module "baseline" --batch_size 1024 --loss "pcc" --is_evaluation 0
python main.py --module "baseline" --batch_size 1024 --loss "pcc" --is_evaluation 1
