# Dual Learning-Based Graph  Neural Network for Remote Sensing Image Super-Resolution
Prepare datasets
===
Prepare your own dataset (HR and LR images) and split it into train/val/test data, the LR images are achieved by bicubic down-sample.

Prepare json file for dataset
===
Get the train.json, val.json and test.json by running ./datasets/make_json_rsdata.py


Train
===
settings in config.py

        __NETWORK.PHASE='train'
        __C.DATASET.DATASET_TRAIN_NAME          = 'rsdata_train'
Run runner.py

Resume
===
settings in config.py

        __NETWORK.PHASE='resume'
        __C.CONST.WEIGHTS = './output/tb_log/DLGNN/checkpoints/best-ckpt.pth'  #model which you want to continue to train
        __C.DATASET.DATASET_TRAIN_NAME          = 'rsdata_train'
Run runner.py

Test
===
settings in config.py

        __NETWORK.PHASE='test'
        __C.CONST.WEIGHTS = './output/tb_log/DLGNN/checkpoints/best-ckpt.pth'  #test model
        __C.DATASET.DATASET_TRAIN_NAME          = 'rsdata_test'
Run runner.py
