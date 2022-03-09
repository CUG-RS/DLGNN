import os
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable, variable

import utils.network_utils as net_utils

from core.val import val
from models.DLGNN import VGG19
from losses.losses import *
from time import time
from torch.autograd import Variable

def train(cfg, init_epoch, train_data_loader, val_data_loader, net, dual_net, solver, lr_scheduler,
          ckpt_dir, train_writer, val_writer, Best_PSNR, Best_Epoch):
    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()

        # Batch average meterics
        batch_time = net_utils.AverageMeter()
        data_time = net_utils.AverageMeter()
        test_time = net_utils.AverageMeter()
        if cfg.TRAIN.USE_PERCET_LOSS: percept_losses = net_utils.AverageMeter()
        pixel_losses = net_utils.AverageMeter()
        total_losses = net_utils.AverageMeter()

        batch_end_time = time()
        n_batches = min(cfg.TRAIN.MAX_INTERS_PER_EPOCH, len(train_data_loader))

        if cfg.TRAIN.USE_PERCET_LOSS:
            vggnet = VGG19()

            if torch.cuda.is_available():
                vggnet = torch.nn.DataParallel(vggnet, range(cfg.CONST.NUM_GPU)).cuda()

        for batch_idx, (_, img_lr_s, img_lr, img_hr) in enumerate(train_data_loader):
            # print('img_lr:',img_lr)
            # print('img_lr_s:',img_lr_s)
            # print('img_hr:',img_hr)

            # set max interrations per epoch
            if batch_idx + 1 > n_batches: break
            # Measure data time
            data_time.update(time() - batch_end_time)
            # Get data from data loader
            img_lr_s, img_lr, img_hr = [net_utils.var_or_cuda(img) for img in [img_lr_s, img_lr, img_hr]]

            # switch models to training mode
            net.train()
            img_out = net(img_lr_s, img_lr)
            sr2lr = dual_net(img_out)
            dual_loss = 0.1*pixelLoss(sr2lr, img_lr, cfg.TRAIN.PIXEL_LOSS)


            pixel_loss = pixelLoss(img_out, img_hr, cfg.TRAIN.PIXEL_LOSS)
            # pixel_loss=(pixelLoss(img_out, img_hr, cfg.TRAIN.PIXEL_LOSS).requires_grad_())
            if cfg.TRAIN.USE_PERCET_LOSS:
                #total_loss=pixel_loss(img_out,img_lr,cfg.TRAIN.PIXEL_LOSS)+0.01 * (perceptualLoss(img_out, img_hr,vggnet))+0.1*(pixelLoss(sr2lr, img_lr, cfg.TRAIN.PIXEL_LOSS))
                percept_loss = 0.01 * perceptualLoss(img_out, img_hr, vggnet)
                #total_loss = (pixelLoss(img_out, img_hr, cfg.TRAIN.PIXEL_LOSS) + 0.01 * perceptualLoss(img_out, img_hr,
                #                                                                                      vggnet)).requires_grad_()
                total_loss=pixel_loss+percept_loss+dual_loss
            else:
                total_loss = Variable(pixelLoss(img_out, img_hr, cfg.TRAIN.PIXEL_LOSS)+0.1*pixelLoss(sr2lr, img_lr, cfg.TRAIN.PIXEL_LOSS),requires_grad=True)
            # total_loss=Variable(total_loss,requires_grad=True)
            # Gradient decent
            solver.zero_grad()
            # total_loss.backward()
            total_loss.backward()
            #print([x.grad for x in solver.param_groups[0]['params']])
            solver.step()
            # Adjust learning rate
            lr_scheduler.step()

            if cfg.TRAIN.USE_PERCET_LOSS:
                percept_losses.update(percept_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            pixel_losses.update(pixel_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)
            total_losses.update(total_loss.item(), cfg.CONST.TRAIN_BATCH_SIZE)

            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar(cfg.CONST.NAME + '/PixelLoss_TRAIN', pixel_loss.item(), n_itr)
            if cfg.TRAIN.USE_PERCET_LOSS:
                train_writer.add_scalar(cfg.CONST.NAME + '/PerceptLoss_TRAIN', percept_loss.item(), n_itr)
            train_writer.add_scalar(cfg.CONST.NAME + '/TotalLoss_TRAIN', total_loss.item(), n_itr)

            # Tick / tock
            batch_time.update(time() - batch_end_time)
            batch_end_time = time()

            if (batch_idx + 1) % cfg.TRAIN.PRINT_FREQ == 0:
                if cfg.TRAIN.USE_PERCET_LOSS:
                    print('[TRAIN] [Ech {0}/{1}][Bch {2}/{3}]\t BT {4}\t DT {5}\t Loss {6} [p: {7}, vgg: {8}]'
                          .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time, data_time,
                                  total_losses, pixel_losses, percept_losses))
                else:
                    print('[TRAIN] [Ech {0}/{1}][Bch {2}/{3}]\t BT {4}\t DT {5}\t Loss {6}'
                          .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, batch_time, data_time,
                                  total_losses))

        # Append epoch loss to TensorBoard
        train_writer.add_scalar(cfg.CONST.NAME + '/TotalLoss_avg_TRAIN', total_losses.avg, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[TRAIN] [Epoch {0}/{1}]\t EpochTime {2}\t TotalLoss_avg {3}\n'
              .format(epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, total_losses.avg))

        # Validate the training models
        if (epoch_idx + 1) % cfg.TRAIN.TEST_FREQ == 0:
            PSNR = val(cfg, epoch_idx, val_data_loader, net, val_writer)

            # Save weights to file
            if PSNR > Best_PSNR:
                Best_PSNR = PSNR
                Best_Epoch = epoch_idx + 1
                net_utils.save_checkpoint(ckpt_dir, 'best-ckpt.pth', epoch_idx + 1, \
                                          net, solver, Best_PSNR, Best_Epoch)
            '''
            if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
                if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
                net_utils.save_checkpoint(ckpt_dir, 'ckpt-epoch-%04d.pth' % (epoch_idx + 1), \
                                          epoch_idx + 1, net, solver, Best_PSNR, Best_Epoch)
            '''
                # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()


