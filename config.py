#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

from easydict import EasyDict as edict
import socket

__C     = edict()
cfg     = __C    #cfg=edict()

#
# Common
#
__C.CONST                               = edict()
__C.CONST.NUM_GPU                       = 1
__C.CONST.NUM_WORKER                    = 0  #4
#__C.CONST.NUM_WORKER                    = 4                     # number of data workers
#__C.CONST.TRAIN_BATCH_SIZE              = 2#2
__C.CONST.TRAIN_BATCH_SIZE             = 1  
__C.CONST.VAL_BATCH_SIZE                = 1
__C.CONST.TEST_BATCH_SIZE               = 1
__C.CONST.NAME                          = 'DLGNNSR'
__C.CONST.WEIGHTS                       = '.'
__C.CONST.SCALE                         = 4

#
# Dataset
#
__C.DATASET                             = edict()

__C.DATASET.DATASET_TRAIN_NAME          = 'rsdata_train'              # rsdata

__C.DATASET.DATASET_TEST_NAME          = 'rsdata_val'              # rsdata_val
__C.DATASET.DATASET_TEST_NAME          = 'rsdata_test'              # rsdata_test

__C.NETWORK                             = edict()
#__C.NETWORK.PHASE                       = 'test'                 # available options: 'train', 'test', 'resume'
__C.NETWORK.PHASE                       = 'train'                # available options: 'train', 'test', 'resume'
#__C.NETWORK.PHASE                       = 'resume'

#
# Directories
#
__C.DIR                                 = edict()
__C.DIR.OUT_PATH = './output'
__C.DIR.DATASET_SCALE                   = 'x'+ str(__C.CONST.SCALE)
__C.DIR.DATASET_ROOT = './' 

if cfg.DATASET.DATASET_TRAIN_NAME == 'rsdata_train':
    __C.DIR.DATASET_JSON_TRAIN_PATH     = './datasets/json_files/RSDataset_train.json'

    #__C.DIR.DATASET_JSON_TRAIN_PATH     = './datasets/json_files/RSDataset_trainx3.json'
    
    #x2 x3
    '''
    __C.DIR.IMAGE_LR_TRAIN_PATH         = __C.DIR.DATASET_ROOT + 'RSDataset/train_lr_bibubic/'+__C.DIR.DATASET_SCALE
    __C.DIR.IMAGE_HR_TRAIN_PATH         = __C.DIR.DATASET_ROOT + 'RSDataset/train_hr_sub/'+__C.DIR.DATASET_SCALE
    '''
    
    #x4
    __C.DIR.IMAGE_LR_TRAIN_PATH = __C.DIR.DATASET_ROOT + 'RSDataset/train_lr_bibubic/' + __C.DIR.DATASET_SCALE
    __C.DIR.IMAGE_HR_TRAIN_PATH = __C.DIR.DATASET_ROOT + 'RSDataset/train_hr/'
    

if cfg.DATASET.DATASET_TRAIN_NAME == 'massets_train':
    __C.DIR.DATASET_JSON_TRAIN_PATH = './datasets/json_files/massets_train_sub.json'
    __C.DIR.IMAGE_LR_TRAIN_PATH         = __C.DIR.DATASET_ROOT + 'massets/train_lr_sub/'+__C.DIR.DATASET_SCALE
    __C.DIR.IMAGE_HR_TRAIN_PATH         = __C.DIR.DATASET_ROOT + 'massets/train_hr_sub'

if cfg.DATASET.DATASET_TEST_NAME == 'rsdata_val':
    __C.DIR.DATASET_JSON_TEST_PATH      = './datasets/json_files/RSDataset_val.json'
    __C.DIR.IMAGE_LR_TEST_PATH          = __C.DIR.DATASET_ROOT + 'RSDataset/val_lr_bibubic/'+__C.DIR.DATASET_SCALE
    __C.DIR.IMAGE_HR_TEST_PATH          = __C.DIR.DATASET_ROOT + 'RSDataset/val_hr'
if cfg.DATASET.DATASET_TEST_NAME == 'rsdata_test':
    __C.DIR.DATASET_JSON_TEST_PATH      = './datasets/json_files/RSDataset_test.json'
    __C.DIR.IMAGE_LR_TEST_PATH          = __C.DIR.DATASET_ROOT + 'RSDataset/test_lr_bibubic/'+__C.DIR.DATASET_SCALE
    __C.DIR.IMAGE_HR_TEST_PATH          = __C.DIR.DATASET_ROOT + 'RSDataset/test_hr'
if cfg.DATASET.DATASET_TEST_NAME == 'ucasaod_val':
    __C.DIR.DATASET_JSON_TEST_PATH     = './datasets/json_files/ucasaod_val_sub.json'
    __C.DIR.IMAGE_LR_TEST_PATH         = __C.DIR.DATASET_ROOT + 'ucasaod/val_lr_sub/'+__C.DIR.DATASET_SCALE
    __C.DIR.IMAGE_HR_TEST_PATH         = __C.DIR.DATASET_ROOT + 'ucasaod/val_hr_sub'
if cfg.DATASET.DATASET_TEST_NAME == 'massets_val':
    __C.DIR.DATASET_JSON_TEST_PATH     = './datasets/json_files/massets_val_sub.json'
    __C.DIR.IMAGE_LR_TEST_PATH         = __C.DIR.DATASET_ROOT + 'massets/val_lr_sub/'+__C.DIR.DATASET_SCALE
    __C.DIR.IMAGE_HR_TEST_PATH         = __C.DIR.DATASET_ROOT + 'massets/val_hr_sub'
if cfg.DATASET.DATASET_TEST_NAME == 'massets_test':
    __C.DIR.DATASET_JSON_TEST_PATH      = './datasets/json_files/massets_test_sub.json'
    __C.DIR.IMAGE_LR_TEST_PATH          = __C.DIR.DATASET_ROOT + 'massets/test_lr_sub/'+__C.DIR.DATASET_SCALE
    __C.DIR.IMAGE_HR_TEST_PATH          = __C.DIR.DATASET_ROOT + 'massets/test_hr_sub'


#
# data augmentation
#
__C.DATA                                = edict()
__C.DATA.RANGE                          = 255
__C.DATA.STD                            = [255.0, 255.0, 255.0]
__C.DATA.MEAN                           = [0.0, 0.0, 0.0]
__C.DATA.GAUSSIAN                       = 9                       # RandomGaussianNoise
__C.DATA.COLOR_JITTER                   = [0.2, 0.15, 0.3, 0.1]   # brightness, contrast, saturation, hue

if cfg.CONST.SCALE == 2: __C.DATA.CROP_IMG_SIZE = [80,80]   #[160,160]
if cfg.CONST.SCALE == 3: __C.DATA.CROP_IMG_SIZE = [198,198]
if cfg.CONST.SCALE == 4: __C.DATA.CROP_IMG_SIZE = [160,160]
#if cfg.CONST.SCALE == 4: __C.DATA.CROP_IMG_SIZE = [40,40]

#
# Network
#
#__C.NETWORK                             = edict()
__C.NETWORK.SRNETARCH                   = 'DLGNN'                  
__C.NETWORK.LEAKY_VALUE                 = 0.0                     # when value = 0.0, lrelu->relu
__C.NETWORK.RES_SCALE                   = 0.1                     # 0.1 for edsr, 1 for baseline edsr
__C.NETWORK.N_RESBLOCK                  = 32  
__C.NETWORK.N_FEATURE                   = 256
__C.NETWORK.N_REIGHBOR                  = 4  
__C.NETWORK.WITH_WINDOW                 = True
__C.NETWORK.WINDOW_SIZE                 = 75  
__C.NETWORK.WITH_ADAIN_NROM             = True
__C.NETWORK.WITH_DIFF                   = True
__C.NETWORK.WITH_SCORE                  = False

# Training
#
__C.TRAIN                               = edict()
__C.TRAIN.PIXEL_LOSS                    = 'L1'                    # available options: 'L1', 'MSE'
__C.TRAIN.USE_PERCET_LOSS               = True #False
__C.TRAIN.NUM_EPOCHES                   = 40*__C.CONST.TRAIN_BATCH_SIZE   #ori=40 maximum number of epoches, bs_2:80, bs_4:160 bs_8:320
__C.TRAIN.MAX_INTERS_PER_EPOCH          = 2000#10000
__C.TRAIN.MAX_INTERS_PER_EPOCH_VAL      = 160#10000   100   120
__C.TRAIN.LEARNING_RATE                 = 1e-4  #1e-4 5e-4
__C.TRAIN.LR_MILESTONES                 = [t*__C.CONST.TRAIN_BATCH_SIZE for t in [8,16,24,32,40]]
__C.TRAIN.LR_DECAY                      = 0.5                     # Multiplicative factor of learning rate decay
__C.TRAIN.MOMENTUM                      = 0.9
__C.TRAIN.BETA                          = 0.999
__C.TRAIN.BIAS_DECAY                    = 0.0                     # regularization of bias, default: 0
__C.TRAIN.WEIGHT_DECAY                  = 0.0                     # regularization of weight, default: 0
__C.TRAIN.KAIMING_SCALE                 = 0.1
__C.TRAIN.PRINT_FREQ                    = 10  #10
__C.TRAIN.SAVE_FREQ                     = 5                       # weights will be overwritten every save_freq epoch
__C.TRAIN.TEST_FREQ                     = 1

#
# Validating options
#
__C.VAL                                 = edict()
__C.VAL.VISUALIZATION_NUM               = 4
__C.VAL.PRINT_FREQ                      = 5  #5

#
# Testing options
#
__C.TEST                                = edict()
__C.TEST.RUNTIME                        = False
__C.TEST.SAVEIMG                        = True
__C.TEST.CHOP                           = True
__C.TEST.ENSEMBLE                       = False
