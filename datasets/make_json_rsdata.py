import json
import os
import io
import numpy as np
samples = []
#root = './RSDataset/train_hr_sub'   #train_sub
root = r'./RSDataset/train_hr_sub/x3'   #train
#root = './RSDataset/val_hr'    #valid
#root = './RSDataset/test_hr'    #valid
#root = './NWPU/nwpu_test_HR'    #test
#root = '.\massets_roads/train_hr_sub'   #train_sub
#root = './massets_roads/train_hr'   #train
#root = './massets_roads/val_hr_sub'    #valid
#root = './massets_roads/test_hr_sub'    #test
sample_list = sorted(os.listdir(root))
sample = [sample_list[i] for i in range(len(sample_list))]
sample_sub = []
for sam in sample:
    if not sam == ".DS_S":
        sample_sub.append(sam)
#train_sub
l = {'name': 'RSDataset', 'phase': 'train','sample': sample_sub}
#l = {'name': 'massets', 'phase': 'train','sample': sample_sub}
#train
#l = {'name': 'RSDataset', 'phase': 'train','sample': sample_sub}
#valid
#l = {'name': 'RSDataset', 'phase': 'test','sample': sample_sub}
#l = {'name': 'massets', 'phase': 'test','sample': sample_sub}
#test
#l = {'name': 'RSDataset', 'phase': 'test','sample': sample_sub}
#l = {'name': 'massets', 'phase': 'test','sample': sample_sub}
samples.append(l)
#filename = './datasets/json_files/RSDataset_train_sub.json'  #train_sub
filename = './datasets/json_files/RSDataset_train.json'  #train
#filename = './datasets/json_files/RSDataset_val.json'  #valid
#filename = './datasets/json_files/RSDataset_test.json'  #test
#filename = './datasets/json_files/massets_train_sub.json'  #train_sub
#filename = './datasets/json_files/massets_train.json'  #train
#filename = './datasets/json_files/massets_val_sub.json'  #valid
#filename = './datasets/json_files/massets_test_sub.json'  #test
with open(filename,'w') as file:
    json.dump(samples,file)