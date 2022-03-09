import os
import sys
import torch.backends.cudnn
import torch.utils.data
import sys
from ptflops import get_model_complexity_info
import numpy as np
from torchsummary import summary
from core.train import train
from core.test import test
from torch.autograd import Variable
sys.path.append(".")
import utils.data_loaders
import utils.data_transforms
import sys
#import runner
from argparse import ArgumentParser
from checkpoint_drn import Checkpoint
#from models.common import dual_model
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
sys.path.append(".")
import utils.network_utils as net_utils
import utility
import model
import models
from tensorboardX import SummaryWriter
#from core.train import train
from core.test import test
from losses.losses import *
from datetime import datetime as dt
#device=torch.device("cpu")
parser = ArgumentParser(description='Parser of Runner of Network')
parser.add_argument('--gpu', dest='gpu_id', help='GPU device to use', default=None, type=str)
parser.add_argument('--phase', dest='phase', help='phase of CNN',default=cfg.NETWORK.PHASE, type=str)
parser.add_argument('--scale', dest='scale', help='factor of upsampling', default=cfg.CONST.SCALE, type=int)
parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=cfg.CONST.WEIGHTS, type=str)
parser.add_argument('--dataset', dest='dataset_path', help='Set dataset root_path', default=cfg.DIR.DATASET_ROOT, type=str)
parser.add_argument('--demodata', dest='demodata_path', help='Set demo test images path',
                    default=cfg.DIR.IMAGE_LR_TEST_PATH, type=str)

parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')  #作用：对图像使用翻转选装等方式进行增广
parser.add_argument('--cpu',
                    default=True,
                    action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--n_blocks', type=int, default=30, # ori=30
                    help='number of residual blocks, 16|30|40|80')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--negval', type=float, default=0.2,
                    help='Negative value parameter for Leaky ReLU')
parser.add_argument('--test_every', type=int, default=20,  #ori=1000
                    help='do test per every N batches')  #每test_every batches test一次
parser.add_argument('--epochs', type=int, default=500,  #ori=1000
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int,
                    #default=32,  #ori
                    #default=4,
                    default=2,
                    help='input batch size for training')
parser.add_argument('--test_only',
                     #default=True,
                    default=False,
                    # action='store_true',  #只要运行时该变量有传参就将该变量设为True
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
                    help='file name to save')  #结果保存的文件夹
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results',
                    default=True,
                    # action='store_true',
                    help='save output results')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--pre_train', type=str,
                    default='.',
                    #default='./pretrained_models/DRNS4x.pt',  # "."
                    #default='./experiment/test/dual_model/model_best27.7.pt',  # "."
                    help='pre-trained dual_model directory')
parser.add_argument('--pre_train_dual', type=str,
                    default='.',  #ori
                    #default='./pretrained_models/DRNS4x_dual_model.pt',
                    #default='./experiment/test/dual_model/dual_model_best27.7.pt',
                    help='pre-trained dual dual_model directory')     #pre_train_dual模型默认路径
args = parser.parse_args()
#print("args_build：",args)
utility.set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device_build:",device)
def bulid_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    # Set up data augmentation
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(cfg.DATA.CROP_IMG_SIZE, cfg.CONST.SCALE),
        utils.data_transforms.FlipRotate(),
        utils.data_transforms.BGR2RGB(),
        utils.data_transforms.RandomColorChannel(),
        utils.data_transforms.ToTensor()
    ])

    test_transforms = utils.data_transforms.Compose([
        # utils.data_transforms.BorderCrop(cfg.CONST.SCALE),
        utils.data_transforms.BGR2RGB(),
        # utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        utils.data_transforms.ToTensor()
    ])

    # Set up data loader
    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_TRAIN_NAME](utils.data_loaders.DatasetType.TRAIN)
    val_dataset_loader=utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_TEST_NAME](utils.data_loaders.DatasetType.TEST)
    test_dataset_loader  = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_TEST_NAME](utils.data_loaders.DatasetType.TEST)
    if cfg.NETWORK.PHASE in ['train', 'resume']:
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset_loader.get_dataset(train_transforms),
            batch_size=cfg.CONST.TRAIN_BATCH_SIZE,#2
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=True)#0
        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_dataset_loader.get_dataset(test_transforms),
            batch_size=cfg.CONST.VAL_BATCH_SIZE,#1
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)
    elif cfg.NETWORK.PHASE in ['test']:
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset_loader.get_dataset(test_transforms),
            batch_size=cfg.CONST.TEST_BATCH_SIZE,#1
            num_workers=cfg.CONST.NUM_WORKER, pin_memory=True, shuffle=False)

    # Set up networks
    net = models.__dict__[cfg.NETWORK.SRNETARCH].__dict__[cfg.NETWORK.SRNETARCH]()   #DLGNN
    
    #print(net)
    print('[DEBUG] %s Parameters in %s: %d.' % (dt.now(), cfg.NETWORK.SRNETARCH,
                                                net_utils.count_parameters(net)))
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, range(cfg.CONST.NUM_GPU)).cuda()
    
    # Initialize weights of networks
    if cfg.NETWORK.PHASE == 'train':
        net_utils.initialize_weights(net, cfg.TRAIN.KAIMING_SCALE)#0.1


    # Set up solver 
    solver = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.TRAIN.LEARNING_RATE,
                                         betas=(cfg.TRAIN.MOMENTUM, cfg.TRAIN.BETA))

    

    # Load pretrained dual_model if exists
    Init_Epoch   = 0
    Best_Epoch   = 0
    Best_PSNR    = 0
    if cfg.NETWORK.PHASE in ['test', 'resume']:  #测试 or 重新开始
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        net.load_state_dict(checkpoint['net_state_dict'])
        if cfg.NETWORK.PHASE == 'resume': Init_Epoch = checkpoint['epoch_idx']
        Best_PSNR  = checkpoint['best_PSNR']
        Best_Epoch = checkpoint['best_epoch']
        if 'solver_state_dict' in checkpoint:
            solver.load_state_dict(checkpoint['solver_state_dict'])

        print('[INFO] {0} Recover complete. Current Epoch #{1}, Best_PSNR = {2} at Epoch #{3}.' \
              .format(dt.now(), Init_Epoch, Best_PSNR, Best_Epoch))

    if cfg.NETWORK.PHASE in ['train', 'resume']:
        # Set up learning rate scheduler to decay learning rates dynamically
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(solver, milestones=cfg.TRAIN.LR_MILESTONES,
                                                                            gamma=cfg.TRAIN.LR_DECAY)
        # Summary writer for TensorBoard
        #output_dir = os.path.join(cfg.DIR.OUT_PATH,'tb_log', dt.now().isoformat()+'_'+cfg.NETWORK.SRNETARCH, '%s')
        output_dir = os.path.join(cfg.DIR.OUT_PATH,'tb_log', cfg.NETWORK.SRNETARCH, '%s')
        log_dir      = output_dir % 'logs'
        ckpt_dir     = output_dir % 'checkpoints'
        train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
        val_writer  = SummaryWriter(os.path.join(log_dir, 'val'))

        utility.set_seed(args.seed)
        checkpoint_drn = Checkpoint(args)

        if checkpoint_drn.ok:
            #loader = data.Data(args)
            model_my = model.Model(args, checkpoint_drn).to(device) #drn  #dual_model:local variable
            #loss = loss.Loss(args, checkpoint) if not args.test_only else None   # if train then loss  else: none
            '''
            t = Trainer(args, model_my,checkpoint,cfg, Init_Epoch, train_data_loader, val_data_loader, net,solver, lr_scheduler, ckpt_dir,
                                                        train_writer, val_writer, Best_PSNR, Best_Epoch)
            

            t = Trainer(args, model_my, checkpoint, cfg, Init_Epoch, train_data_loader, val_data_loader, net,
                        lr_scheduler, ckpt_dir,
                        train_writer, val_writer, Best_PSNR, Best_Epoch)
            '''
            #net:DLGNN
            #t.train()
            dual_net=model_my.dual_model.cuda()
            # train and val
            train(cfg, Init_Epoch, train_data_loader, val_data_loader, net, dual_net, solver, lr_scheduler, ckpt_dir,
                  train_writer, val_writer, Best_PSNR, Best_Epoch)

        return
    elif cfg.NETWORK.PHASE in ['test']:
        if cfg.DATASET.DATASET_TEST_NAME == 'rsdata_test':  
            with torch.no_grad():
                test(cfg, test_data_loader, net, Best_Epoch)  
        return