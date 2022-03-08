# Dual Learning-Based Graph  Neural Network for Remote Sensing Image Super-Resolution
Prepare datasets
===
Prepare your own dataset (HR and LR images) and split it into train/val/test data, the LR images are achieved by bicubic down-sample.

Train
===
Use the following command to train the network:
```
python runner.py
        --gpu [gpu_id]\
        --phase 'train'\
        --scale [2/3/4]\
        --dataroot [dataset root]\
        --out [output path]
```

Use the following command to resume training the network:
```
python runner.py 
        --gpu [gpu_id]\
        --phase 'resume'\
        --weights './ckpt/IGNN_x[2/3/4].pth'\
        --scale [2/3/4]\
        --dataroot [dataset root]\
        --out [output path]
```
You can also use the following simple command with different settings in config.py:
```
python runner.py
```

Test
===
Use the following command to test the network:
```
python runner.py \
        --gpu [gpu_id]\
        --phase 'test'\
        --weights './ckpt/IGNN_x[2/3/4].pth'\
        --scale [2/3/4]\
        --demopath [test folder path]\
        --testname 'Demo'\
        --out [output path]
```
