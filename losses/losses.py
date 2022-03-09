import torch
import torch.nn as nn
from config import cfg
#device=torch.device("cpu")

def pixelLoss(output, target, type='L1'):
    if output.shape != target.shape:
        output1 = output[:, :, :min(output.shape[2], target.shape[2]), :min(output.shape[3], target.shape[3])]
    else:
        output1=output
    if type == 'L1':
        loss = nn.L1Loss()
    if type == 'MSE':
        loss = nn.MSELoss()
    return loss(output1, target)


def perceptualLoss(output, target, vggnet):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''
    weights = [1, 0.2, 0.04]
    features_fake = vggnet(output)  #fakeIm
    features_real = vggnet(target)  #realIm
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    mse_loss = nn.MSELoss()

    loss = 0
    for i in range(len(features_real)):
        #print(features_fake[i].shape)
        #print(features_real_no_grad[i].shape)

        if features_fake[i].shape != features_real_no_grad[i].shape:
            features_fake[i] = features_fake[i][:,:,:min(features_fake[i].shape[2],features_real_no_grad[i].shape[2]),:min(features_fake[i].shape[3],features_real_no_grad[i].shape[3])]
            #print("features_fake[i].shape:", features_fake[i].shape)
            loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        else:
            loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss
