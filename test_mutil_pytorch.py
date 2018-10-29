import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
import torch.utils.data
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
import cv2
import json
import numpy as np

from model.configAli import cfg
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from utils.imutils import im_to_numpy, im_to_torch
from networks import network_dr
from dataloader.aliDataset import MscocoMulti
from tqdm import tqdm
import csv
from Evaluator import FaiKeypoint2018Evaluator

scales = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
# scales=[1,0.9]
def main(args):
    # create model
    model = network_dr.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=False)
    model = torch.nn.DataParallel(model).cuda()



    # load trainning weights
    # checkpoint_file = os.path.join(args.checkpoint, args.test + '.pth.tar')
    checkpoint_file = os.path.join('model', 'checkpoint', 'epoch16checkpoint_dr_newlr.pth.tar')
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    # change to evaluation mode
    model.eval()
    print('mutil_testing...')
    for scale in scales:
        full_result = []
        test_loader = torch.utils.data.DataLoader(
            MscocoMulti(cfg,scale,train=False),
            batch_size=args.batch * args.num_gpus, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        for i, (inputs, meta) in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                input_var = torch.autograd.Variable(inputs.cuda())
                if args.flip == True:
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

                if args.flip == True:
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
        result_path = args.result
        if not isdir(result_path):
            mkdir_p(result_path)
        result_file = os.path.join(result_path, hzx+str(scale)+'.csv')
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(full_result)


    output_dir=args.result
    for scale in scales:
        cur_file = hzx + str(scale) + ".csv"
        with open(os.path.join(output_dir, cur_file), "r") as f:
            tmp = []
            for line in f.readlines():
                tmp.append(line)
            locals()["tmp_res_" + str(scale)] = tmp

    res_multi_file = hzx + "multiNEWGAME_45.csv"
    with open(os.path.join(output_dir, res_multi_file), "w", newline="") as f:
        writer = csv.writer(f)
        for i in range(len(locals()["tmp_res_1"])):
            kp_multi = [0] * 48
            for scale in scales:
                locals()["info_" + str(scale)] = locals()["tmp_res_" + str(scale)][i]
                locals()["kp_" + str(scale)] = locals()["info_" + str(scale)].split(",")[2:]
                for idx, kp in enumerate(locals()["kp_" + str(scale)]):
                    # idx = locals()["kp_" + str(scale)].index(kp)
                    x, y, _ = kp.split("_")
                    kp_multi[idx * 2] += float(x)
                    kp_multi[idx * 2 + 1] += float(y)
            base = locals()["info_1"].split(',')[:2]

            kp_multi = [t / len(scales) for t in kp_multi]
            kp_multi = [round(t) for t in kp_multi]

            j = 0
            kp_res = []
            while j < len(kp_multi):
                kp_res.append(str(kp_multi[j]) + '_' + str(kp_multi[j + 1]) + '_1')
                j += 2

            out = base + kp_res
            writer.writerow(out)



    Evaluator = FaiKeypoint2018Evaluator(userAnswerFile=os.path.join(output_dir, res_multi_file),
                                         standardAnswerFile="fashionAI_key_points_test_a_answer_20180426.csv")
    score = Evaluator.evaluate()

    print(score)

    # Evaluator.writerror(result_path=os.path.join(result_path, "toperror1.csv"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Test')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint)')
    parser.add_argument('-f', '--flip', default=True, type=bool,
                        help='flip input image during test (default: True)')
    parser.add_argument('-b', '--batch', default=4, type=int,
                        help='test batch size (default: 128)')
    parser.add_argument('-t', '--test', default='CPN384x288', type=str,
                        help='using which checkpoint to be tested (default: CPN256x192')
    parser.add_argument('-r', '--result', default='result_mutil', type=str,
                        help='path to save save result (default: result)')
    hzx='MutilresultNEWGAME45_'
    main(parser.parse_args())
