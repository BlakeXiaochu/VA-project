from __future__ import print_function
import argparse
import random
import time
import os
import numpy as np
from optparse import OptionParser

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR as LR_Policy

import models
from dataset import VideoFeatDataset as dset
from tools.config_tools import Config
from tools import utils


# load opts
parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="training configuration",
                  default="./configs/train_config.yaml")

(train_opts, args) = parser.parse_args()
assert isinstance(train_opts, object)
train_opt = Config(train_opts.config)
print(train_opt)

parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/test_config.yaml")

(test_opts, args) = parser.parse_args()
assert isinstance(test_opts, object)
test_opt = Config(test_opts.config)
print(test_opt)

if train_opt.checkpoint_folder is None:
    train_opt.checkpoint_folder = 'checkpoints'
if test_opt.checkpoint_folder is None:
    test_opt.checkpoint_folder = 'checkpoints'

# make dir
if not os.path.exists(train_opt.checkpoint_folder):
    os.system('mkdir {0}'.format(train_opt.checkpoint_folder))

# load dataset
train_dataset = dset(train_opt.data_dir, flist=train_opt.flist)

print('number of train samples is: {0}'.format(len(train_dataset)))
print('finished loading data')

test_video_dataset = dset(test_opt.data_dir, test_opt.video_flist, which_feat='vfeat')
test_audio_dataset = dset(test_opt.data_dir, test_opt.audio_flist, which_feat='afeat')

print('number of test samples is: {0}'.format(len(test_video_dataset)))
print('finished loading data')


if train_opt.manualSeed is None:
    train_opt.manualSeed = random.randint(1, 10000)

if torch.cuda.is_available() and not train_opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \"cuda: True\"")
    torch.manual_seed(train_opt.manualSeed)
else:
    if int(train_opt.ngpu) == 1:
        print('so we use 1 gpu to training')
        print('setting gpu on gpuid {0}'.format(train_opt.gpu_id))

        if train_opt.cuda:
            torch.cuda.set_device(int(train_opt.gpu_id) - 1)
            os.environ['CUDA_VISIBLE_DEVICES'] = train_opt.gpu_id
            torch.cuda.manual_seed(train_opt.manualSeed)
            cudnn.benchmark = True
print('Random Seed: {0}'.format(train_opt.manualSeed))


# test function for metric learning
def test(video_loader, audio_loader, model, test_opt):
    # test mode
    model.eval()

    right = [0] * test_opt.topk
    for _, vfeat in enumerate(video_loader):
        for _, afeat in enumerate(audio_loader):
            # shuffling the index orders
            bz = vfeat.size()[0]
            simmat = 0
            for k in np.arange(bz):
                cur_vfeat = vfeat[k, :, :].clone()
                cur_vfeats = cur_vfeat.repeat(bz, 1, 1)

                vfeat_var = Variable(cur_vfeats)
                afeat_var = Variable(afeat)

                vfeat_var.volatile = True
                afeat_var.volatile = True

                if train_opt.cuda:
                    vfeat_var = vfeat_var.cuda()
                    afeat_var = afeat_var.cuda()

                cur_sim = model.forward(vfeat_var, afeat_var, stage='TEST')
                cur_sim = cur_sim.view(cur_sim.size(0), -1)
                if k == 0:
                    simmat = cur_sim.clone()
                else:
                    simmat = torch.cat((simmat, cur_sim), 1)
            _, indices = torch.sort(simmat, 0)
            np_indices = indices.cpu().data.numpy()
            topk = np_indices[:test_opt.topk,:]
            for k in np.arange(bz):
                order = topk[:,k]
                for j in range(test_opt.topk):
                    if k in order[:j+1]:
                        right[j] = right[j] + 1
            # print('The similarity matrix: \n {}'.format(simmat))
            right = [i/bz for i in right]
            print('Testing accuracy (top{}): {}'.format(test_opt.topk, right))


# training function for metric learning
def train(train_loader, model, criterion, optimizer, epoch, train_opt):
    """
    train for one epoch on the training set
    """
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # training mode
    model.train()

    end = time.time()

    for i, (vfeat0, afeat0) in enumerate(train_loader):
        # shuffle 1: pairs shuffle
        bz = vfeat0.size(0)
        p_orders = np.arange(bz).astype('int32')
        shuffle_orders1 = p_orders.copy()
        np.random.shuffle(shuffle_orders1)
        shuffle_orders2 = p_orders.copy()
        np.random.shuffle(shuffle_orders2)

        afeat1 = afeat0[torch.from_numpy(shuffle_orders1).long()].clone()
        afeat2 = afeat0[torch.from_numpy(shuffle_orders2).long()].clone()
        vfeat1 = vfeat0
        vfeat2 = vfeat0

        # shuffle 2: time shuffle
        # tl = vfeat0.size(1)
        # t_orders = np.arange(tl).astype('int32')
        # shuffle_orders2 = t_orders.copy()
        # np.random.shuffle(shuffle_orders2)
        # afeat2 = afeat0[:, torch.from_numpy(shuffle_orders2).long(), :].clone()
        # vfeat2 = vfeat0

        # concat data
        afeat = torch.cat((afeat0, afeat1, afeat2), 0)
        vfeat = torch.cat((vfeat0, vfeat1, vfeat2), 0)

        # generating the labels
        # 1. the labels for the shuffled feats
        label1 = (p_orders == shuffle_orders1 + 0).astype('float32')
        target1 = torch.from_numpy(label1).long()
        # label2 = (np.zeros(bz)).astype('float32')
        label2 = (p_orders == shuffle_orders2 + 0).astype('float32')
        target2 = torch.from_numpy(label2).long()

        # 2. the labels for the original feats
        label0 = label1.copy()
        label0[:] = 1
        target0 = torch.from_numpy(label0).long()

        # concat the labels together
        target = torch.cat((target0, target1, target2), 0)

        # put the data into Variable
        vfeat_var = Variable(vfeat)
        afeat_var = Variable(afeat)
        target_var = Variable(target)

        # if you have gpu, then shift data to GPU
        if train_opt.cuda:
            vfeat_var = vfeat_var.cuda()
            afeat_var = afeat_var.cuda()
            target_var = target_var.cuda()

        # forward, backward optimize
        sim = model.forward(vfeat_var, afeat_var, stage='TRAIN')   # inference simialrity
        loss = criterion.forward(sim, target_var)   # compute contrastive loss

        losses.update(loss.data[0], vfeat0.size(0))

        optimizer.zero_grad()
        loss.backward()
        utils.clip_gradient(optimizer, train_opt.gradient_clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % train_opt.print_freq == 0:
            log_str = 'Epoch: [{0}][{1}/{2}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses)
            print(log_str)


def main():
    global train_opt
    # train data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_opt.batchSize,
                                               shuffle=True, num_workers=int(train_opt.workers))
    global test_opts
    # test data loader
    test_video_loader = torch.utils.data.DataLoader(test_video_dataset, batch_size=test_opt.batchSize,
                                                    shuffle=False, num_workers=int(test_opt.workers))
    test_audio_loader = torch.utils.data.DataLoader(test_audio_dataset, batch_size=test_opt.batchSize,
                                                    shuffle=False, num_workers=int(test_opt.workers))

    # create model
    model = models.VAMetric()

    if train_opt.init_model != '':
        print('loading pretrained model from {0}'.format(train_opt.init_model))
        if train_opt.cuda:
            model.load_state_dict(torch.load(train_opt.init_model, map_location=lambda storage, loc: storage.cuda(int(train_opt.gpu_id) - 1)))
        else:
            model.load_state_dict(torch.load(train_opt.init_model, map_location=lambda storage, loc: storage))

    # Contrastive Loss
    criterion = models.ContrastiveLoss()

    if train_opt.cuda:
        print('shift model and criterion to GPU .. ')
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), weight_decay=train_opt.weight_decay)

    for epoch in range(train_opt.max_epochs):
        #################################
        # train for one epoch
        #################################
        train(train_loader, model, criterion, optimizer, epoch, train_opt)
        # scheduler.step()

        # testing
        if ((epoch + 1) % train_opt.test_epochs) == 0:
            test(test_video_loader, test_audio_loader, model, test_opt)

        ##################################
        # save checkpoint every train_opt.epoch_save epochs
        ##################################
        if ((epoch + 1) % train_opt.epoch_save) == 0:
            path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format(train_opt.checkpoint_folder, train_opt.prefix, epoch + 1)
            utils.save_checkpoint(model.state_dict(), path_checkpoint)

if __name__ == '__main__':
    main()
