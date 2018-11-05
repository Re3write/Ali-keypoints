import os
import argparse
import time
import matplotlib.pyplot as plt
import torch.utils.data
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(proj_dir)
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import numpy as np
# from 384.288
from model.configAli import cfg
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from networks import network_dr
from dataloader.aliDataset import MscocoMulti
from tqdm import tqdm
from utils.imutils import im_to_numpy, im_to_torch
import cv2
import csv
from Evaluator import FaiKeypoint2018Evaluator

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

def main(args):
    # create checkpoint dir
    # print(torch.cuda.device_count())
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    model = network_dr.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=True)
    model = torch.nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()

    # define loss function (criterion) and optimizer
    criterion1 = torch.nn.MSELoss().cuda()  # for Global loss
    criterion2 = torch.nn.MSELoss(reduce=False).cuda()  # for refine loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    cudnn.benchmark = True
    print('    Total params: %.2fMB' % (sum(p.numel() for p in model.parameters()) / (1024 * 1024) * 4))

    train_loader = torch.utils.data.DataLoader(
        MscocoMulti(cfg,scale=10000),
        batch_size=cfg.batch_size * args.num_gpus, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    trainRecordloss=1000000
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        log_file = []
        # train for one epoch
        train_loss,score = train(train_loader, model, [criterion1, criterion2], optimizer,epoch)
        print('train_loss: ', train_loss)
        if trainRecordloss-train_loss<0.15:
           for param_group in optimizer.param_groups:
               param_group['lr'] *= 0.5

        if train_loss<trainRecordloss:
            trainRecordloss=train_loss

        # append logger file
        # logger.append([epoch + 1, lr, train_loss])

        log_file.append([epoch,train_loss,score,lr])
        save_model({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint=args.checkpoint)

        with open('log_file.csv', 'a', newline='') as f:
            writer1=csv.writer(f)
            writer1.writerows(log_file)
    # logger.close()


def train(train_loader, model, criterions, optimizer,epoch):
    # prepare for refine loss
    def ohkm(loss, top_k):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / top_k
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    criterion1, criterion2 = criterions

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for i, (inputs, targets, valid, meta) in enumerate(train_loader):
        input_var = torch.autograd.Variable(inputs.cuda())

        target15, target11, target9, target7 = targets
        refine_target_var = torch.autograd.Variable(target7.cuda(async=True))
        valid_var = torch.autograd.Variable(valid.cuda(async=True))

        # compute output
        global_outputs, refine_output = model(input_var)
        score_map = refine_output.data.cpu()

        loss = 0.
        global_loss_record = 0.
        refine_loss_record = 0.
        # comput global loss and refine loss
        for global_output, label in zip(global_outputs, targets):
            num_points = global_output.size()[1]
            global_label = label * (valid > 0.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            global_loss = criterion1(global_output, torch.autograd.Variable(global_label.cuda(async=True))) / 2.0
            loss += global_loss
            global_loss_record += global_loss.data.item()
        refine_loss = criterion2(refine_output, refine_target_var)
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        refine_loss *= (valid_var > -0.9).type(torch.cuda.FloatTensor)
        refine_loss = ohkm(refine_loss, 8)
        loss += refine_loss
        refine_loss_record = refine_loss.data.item()

        # record loss
        losses.update(loss.data.item(), inputs.size(0))

        # compute gradient and do Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % 100 == 0 and i != 0):
            print('iteration {} | loss: {}, global loss: {}, refine loss: {}, avg loss: {}'
                  .format(i, loss.data.item(), global_loss_record,
                          refine_loss_record, losses.avg))

    # change to evaluation mode
    model.eval()
    flip=True

    test_loader = torch.utils.data.DataLoader(
        MscocoMulti(cfg, scale=0, train=False),
        batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True)

    # load trainning weights
    # checkpoint_file = os.path.join(args.checkpoint, args.test + '.pth.tar')
    print('testing...')
    full_result = []
    for i, (inputs, meta) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inputs.cuda())
            if flip == True:
                flip_inputs = inputs.clone()
                for i, finp in enumerate(flip_inputs):
                    finp = im_to_numpy(finp)
                    finp = cv2.flip(finp, 1)
                    flip_inputs[i] = im_to_torch(finp)
                flip_input_var = torch.autograd.Variable(flip_inputs.cuda())

            # compute output
            global_outputs, refine_output = model(input_var)
            score_map = refine_output.data.cpu()
            score_map = score_map.numpy()

            if flip == True:
                flip_global_outputs, flip_output = model(flip_input_var)
                flip_score_map = flip_output.data.cpu()
                flip_score_map = flip_score_map.numpy()

                for i, fscore in enumerate(flip_score_map):
                    fscore = fscore.transpose((1, 2, 0))
                    fscore = cv2.flip(fscore, 1)
                    fscore = list(fscore.transpose((2, 0, 1)))
                    for (q, w) in cfg.symmetry:
                        fscore[q], fscore[w] = fscore[w], fscore[q]
                    fscore = np.array(fscore)
                    score_map[i] += fscore
                    score_map[i] /= 2

            # ids = meta['imgID'].numpy()
            det_scores = meta['det_scores']
            for b in range(inputs.size(0)):
                details = meta['augmentation_details']
                imgid = meta['imgid'][b]
                # print(imgid)
                category = meta['category'][b]
                # print(category)
                single_result_dict = {}
                single_result = []

                single_map = score_map[b]
                r0 = single_map.copy()
                r0 /= 255
                r0 += 0.5
                v_score = np.zeros(24)
                for p in range(24):
                    single_map[p] /= np.amax(single_map[p])
                    border = 10
                    dr = np.zeros((cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
                    dr[border:-border, border:-border] = single_map[p].copy()
                    dr = cv2.GaussianBlur(dr, (21, 21), 0)
                    lb = dr.argmax()
                    y, x = np.unravel_index(lb, dr.shape)
                    dr[y, x] = 0
                    lb = dr.argmax()
                    py, px = np.unravel_index(lb, dr.shape)
                    y -= border
                    x -= border
                    py -= border + y
                    px -= border + x
                    ln = (px ** 2 + py ** 2) ** 0.5
                    delta = 0.25
                    if ln > 1e-3:
                        x += delta * px / ln
                        y += delta * py / ln
                    x = max(0, min(x, cfg.output_shape[1] - 1))
                    y = max(0, min(y, cfg.output_shape[0] - 1))
                    resy = float((4 * y + 2) / cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                    resx = float((4 * x + 2) / cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                    v_score[p] = float(r0[p, int(round(y) + 1e-10), int(round(x) + 1e-10)])
                    single_result.append(resx)
                    single_result.append(resy)
                    single_result.append(1)
                if len(single_result) != 0:
                    result = []
                    result.append(imgid)
                    result.append(category)
                    j = 0
                    while j < len(single_result):
                        result.append(str(int(single_result[j])) + '_' + str(int(single_result[j + 1])) + '_1')
                        j += 3
                    full_result.append(result)

    result_path = 'train_log'
    if not isdir(result_path):
        mkdir_p(result_path)
    result_file = os.path.join(result_path, 'result101_{}_dr45_se.csv'.format(epoch))
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(full_result)

    Evaluator = FaiKeypoint2018Evaluator(userAnswerFile=os.path.join(result_path, 'result101_{}_dr45_se.csv'.format(epoch)),
                                         standardAnswerFile="fashionAI_key_points_test_a_answer_20180426.csv")
    score = Evaluator.evaluate()

    print(score)

    return losses.avg,score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('--epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='False', type=str, metavar='PATH',
                        help='path to latest checkpoint')

    main(parser.parse_args())
