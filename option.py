import argparse
import utility
import numpy as np
import torch

parser = argparse.ArgumentParser(description='dl')

parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', 
                    default=True,
                    action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--data_dir', type=str, default='./data_path',
                    help='dataset directory')
parser.add_argument('--data_train', type=str, default='DF2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set14',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--scale', type=int, default=4,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, 
                    #default=80,
                    default=384,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--dual_model', default="dl-S", help='dual_model name,dl-S | dl-L',
                    # required=True
                    )
parser.add_argument('--pre_train', type=str,
                    default='.',
                    #default='./pretrained_models/DRNS4x.pt',  # "."
                    #default='./experiment/test/dual_model/model_best.pt',  # "."
                    help='pre-trained dual_model directory')
parser.add_argument('--pre_train_dual', type=str,
                    default='.',  #ori
                    #default='./pretrained_models/DRNS4x_dual_model.pt',
                    #default='./experiment/test/dual_model/dual_model_best.pt',
                    help='pre-trained dual dual_model directory')
parser.add_argument('--n_blocks', type=int, default=30, # ori=30
                    help='number of residual blocks, 16|30|40|80')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--negval', type=float, default=0.2, 
                    help='Negative value parameter for Leaky ReLU')
parser.add_argument('--test_every', type=int, default=20,  #ori=1000
                    help='do test per every N batches')  #
parser.add_argument('--epochs', type=int, default=500,  #ori=1000
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, 
                    #default=32,  #ori
                    #default=4,
                    default=2,
                    help='input batch size for training')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only',
                     #default=True,
                    default=False,
                    # action='store_true', 
                    help='set this option to test the dual_model')
parser.add_argument('--lr', type=float,
                    default=1e-4,
                    #default=1e-5,
                    help='learning rate')
parser.add_argument('--eta_min', type=float, default=1e-7,
                    help='eta_min lr')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration, L1|MSE')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')
parser.add_argument('--dual_weight', type=float, default=0.1,
                    help='the weight of dual loss')
parser.add_argument('--save', type=str, default='./experiment/test/',
                    help='file name to save')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', 
                    default=True,
                    # action='store_true',
                    help='save output results')

args = parser.parse_args()
print("args.seed:",args.seed)
utility.set_seed(args.seed)
print("n_threads:",args.n_threads)
print("dual_model:",args.model)
print("pre_train:",args.pre_train)
print("pre_train_dual:",args.pre_train_dual)
print("cpu:",args.cpu)
print("data_dir:",args.data_dir)
print("data_train:",args.data_train)
print("patch_size:",args.patch_size)
print("epochs:",args.epochs)
print(":",args.cpu)
print(":",args.cpu)

utility.init_model(args)

# scale = [2,4] for 4x SR to load data
# scale = [2,4,8] for 8x SR to load data
args.scale = [pow(2, s+1) for s in range(int(np.log2(args.scale)))]

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
