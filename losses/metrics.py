'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import copy
import numpy as np
import cv2
import torch
#device=torch.device("cpu")

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    #print("img1.type:",type(img1))
    #print("img1.shape:",img1.shape)
    #print("img2.shape:", img2.shape)
    #img1=img1[:min(img1.shape[0],img2.shape[0]),:min(img1.shape[1],img2.shape[1])]
    #img2=img2[:min(img1.shape[0],img2.shape[0]),:min(img1.shape[1],img2.shape[1])]
    diff_0 = img1.shape[0] - img2.shape[0]
    diff_1 = img1.shape[1] - img2.shape[1]
    if diff_0 > 0:  # img1行数大于img2行数
        if diff_1 == 0:
            # print(img1[-diff_0:,:])
            # img1=img1[:img2.shape[0],:]
            img2 = np.concatenate((img2, img1[-diff_0:, :]), axis=0)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)

        if diff_1 > 0:  # img1列数大于img2列数
            # img1=img1[:img2.shape[0],:img2.shape[1]]
            img2 = np.concatenate((img2, img1[-diff_0:, :img2.shape[1]]), axis=0)
            img2 = np.concatenate((img2, img1[:, -diff_1:]), axis=1)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)
        if diff_1 < 0:  # img1列数小于img2列数
            # img1=img1[:img2.shape[0],:img1.shape[1]]
            # img2=img2[:,:img1.shape[1]]
            img2 = img2[:, :img1.shape[1]]
            img2 = np.concatenate((img2, img1[diff_0:, :]), axis=0)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)

    if diff_0 < 0:  # img1行数小于img2行数
        if diff_1 == 0:  # img1列数等于img2列数
            # print(img1[-diff_0:,:])
            # img1=img1[:img2.shape[0],:]
            img1 = np.concatenate((img1, img2[diff_0:, :]), axis=0)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)

        if diff_1 > 0:  # img1列数大于img2列数
            # img1=img1[:img2.shape[0],:img2.shape[1]]
            img1 = img1[:, :img2.shape[1]]
            img1 = np.concatenate((img1, img2[diff_0:, :]), axis=0)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)
        if diff_1 < 0:  # img1列数小于img2列数
            # img1=img1[:img2.shape[0],:img1.shape[1]]
            # img2=img2[:,:img1.shape[1]]
            img1 = np.concatenate((img1, img2[diff_0:, :img1.shape[1]]), axis=0)
            img1 = np.concatenate((img1, img2[:, diff_1:]), axis=1)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)
    if diff_0 == 0:
        if diff_1 == 0:
            img1 = img1
            img2 = img2
        if diff_1 > 0:
            img2 = np.concatenate((img2, img1[:, -diff_1:]), axis=1)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)
        if diff_1 < 0:
            img1 = np.concatenate((img1, img2[:, diff_1:]), axis=1)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1[:min(img1.shape[0], img2.shape[0]), :min(img1.shape[1], img2.shape[1])]
    #img2 = img2[:min(img1.shape[0], img2.shape[0]), :min(img1.shape[1], img2.shape[1])]
    diff_0 = img1.shape[0] - img2.shape[0]
    diff_1 = img1.shape[1] - img2.shape[1]
    if diff_0 > 0:  # img1行数大于img2行数
        if diff_1 == 0:
            # print(img1[-diff_0:,:])
            # img1=img1[:img2.shape[0],:]
            img2 = np.concatenate((img2, img1[-diff_0:, :]), axis=0)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)

        if diff_1 > 0:  # img1列数大于img2列数
            # img1=img1[:img2.shape[0],:img2.shape[1]]
            img2 = np.concatenate((img2, img1[-diff_0:, :img2.shape[1]]), axis=0)
            img2 = np.concatenate((img2, img1[:, -diff_1:]), axis=1)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)
        if diff_1 < 0:  # img1列数小于img2列数
            # img1=img1[:img2.shape[0],:img1.shape[1]]
            # img2=img2[:,:img1.shape[1]]
            img2 = img2[:, :img1.shape[1]]
            img2 = np.concatenate((img2, img1[diff_0:, :]), axis=0)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)

    if diff_0 < 0:  # img1行数小于img2行数
        if diff_1 == 0:  # img1列数等于img2列数
            # print(img1[-diff_0:,:])
            # img1=img1[:img2.shape[0],:]
            img1 = np.concatenate((img1, img2[diff_0:, :]), axis=0)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)

        if diff_1 > 0:  # img1列数大于img2列数
            # img1=img1[:img2.shape[0],:img2.shape[1]]
            img1 = img1[:, :img2.shape[1]]
            img1 = np.concatenate((img1, img2[diff_0:, :]), axis=0)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)
        if diff_1 < 0:  # img1列数小于img2列数
            # img1=img1[:img2.shape[0],:img1.shape[1]]
            # img2=img2[:,:img1.shape[1]]
            img1 = np.concatenate((img1, img2[diff_0:, :img1.shape[1]]), axis=0)
            img1 = np.concatenate((img1, img2[:, diff_1:]), axis=1)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)
    if diff_0 == 0:
        if diff_1 == 0:
            img1 = img1
            img2 = img2
        if diff_1 > 0:
            img2 = np.concatenate((img2, img1[:, -diff_1:]), axis=1)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)
        if diff_1 < 0:
            img1 = np.concatenate((img1, img2[:, diff_1:]), axis=1)
            #print("img1:", img1)
            #print("img2:", img2)
            #print("img1.shape:", img1.shape)
            #print("img2.shape:", img2.shape)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


