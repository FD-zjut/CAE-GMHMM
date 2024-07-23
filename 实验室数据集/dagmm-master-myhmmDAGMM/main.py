import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *
np.random.seed(20)
torch.manual_seed(10)

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    mkdir(config.log_path)
    mkdir(config.model_save_path)

    data_loader = get_loader(config.data_path, batch_size=config.batch_size, mode=config.mode)
    print("data_loadercc:",data_loader)
    iters_per_epochcc = len(data_loader)
    print("iters_per_epochcc:",iters_per_epochcc)
    # Solver
    solver = Solver(data_loader, vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-4)


    # Training settings
    parser.add_argument('--num_epochs', type=int, default=10)#200
    parser.add_argument('--batch_size', type=int, default=5)#1024
    parser.add_argument('--gmm_k', type=int, default=6) #4
    parser.add_argument('--latent_dim', type=int, default=10)  # 3 应该是大于等于7
    parser.add_argument('--num_states', type=int, default=5)  # 3

    parser.add_argument('--lambda_energy', type=float, default=0.000001)#0.1
    parser.add_argument('--lambda_cov_diag', type=float, default=1e-9)#0.005
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Path
    # parser.add_argument('--data_path', type=str, default='kdd_cup.npz')#kdd_cup #data_no_gwf
    # parser.add_argument('--data_path', type=str, default='data_no_gwf.txt')  # kdd_cup #data_no_gwf#cw
    parser.add_argument('--data_path', type=str, default='data.txt')  # kdd_cup #data_no_gwf#cw
    parser.add_argument('--log_path', type=str, default='./dagmm/logs')
    parser.add_argument('--model_save_path', type=str, default='./dagmm/models')

    # Step size
    parser.add_argument('--log_step', type=int, default=1)#10
    parser.add_argument('--sample_step', type=int, default=194)
    parser.add_argument('--model_save_step', type=int, default=194)

    config = parser.parse_args()
 
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)