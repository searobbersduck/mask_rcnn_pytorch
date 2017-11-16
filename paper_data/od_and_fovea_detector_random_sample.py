# this file to detect optic-disc and fovea

import os
import sys
sys.path.append('../')

import torch.nn as nn
import math
import torchvision

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, Scale
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable

import torch.optim as optim

# from data_xml import DRDetectionDS_xml, coco_collate

from torch.utils.data import Dataset, DataLoader
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from glob import glob

from utils import AverageMeter
import time

import argparse
import torch.backends.cudnn as cudnn

from utils import PILColorJitter, Lighting

from predict_common import DRDetection_predict_raw, pts_trans_inv, scale_image, \
    get_detect_od_array, get_random_bbox, DRDetectionDS_od_and_fovea_xml, DRDetectionDS_od_and_fovea_predict_xml

from detection_common_model import cls_model_od_and_fovea, cls_train, cls_val

import random

def parse_args():
    parser = argparse.ArgumentParser(description='multi-task classification options')
    # parser.add_argument('--root', required=True)
    parser.add_argument('--traincsv', default=None)
    parser.add_argument('--valcsv', default=None)
    parser.add_argument('--testcsv', default=None)
    parser.add_argument('--exp', default='od_and_fovea_detection', help='The name of experiment')
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--crop', default=512, type=int)
    parser.add_argument('--size', default=512, type=int)
    parser.add_argument('--weight', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--epoch', default=10000, type=int)
    parser.add_argument('--fix', default=1000, type=int)
    parser.add_argument('--step', default=2000, type=int)
    parser.add_argument('--dataset', default='emei', choices=['emei', 'wudang'])
    parser.add_argument('--model', default='rsn34', choices=[
        'rsn18', 'rsn34', 'rsn50', 'rsn101', 'rsn150', 'dsn121', 'dsn161', 'dsn169', 'dsn201',
    ])
    parser.add_argument('--seed', default=111, type=int)
    parser.add_argument('--phase', default='train', choices=['train', 'test', 'infer'])
    parser.add_argument('--display', default=100, type=int)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--output', default='output', help='The output dir')

    parser.add_argument('--infer_root', default=None)
    parser.add_argument('--dme_weight_aug', default=1.0, type=float)

    return parser.parse_args()

def cls_eval(val_data_loader, model, criterion, display):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    logger = []
    trans = ToPILImage()
    for num_iter, (images, anns, image_paths, bboxs, bboxs_c) in enumerate(val_data_loader):
        data_time.update(time.time() - end)
        # bbox_od = Variable(anns[0]['bbox'].type(torch.FloatTensor).cuda())
        # bbox_od = Variable(bboxs.type(torch.FloatTensor).cuda())
        bbox_od_c = Variable(bboxs_c.type(torch.FloatTensor).cuda())
        final, map = model(Variable(images))
        # loss = criterion(final, bbox_od)
        loss = criterion(final, bbox_od_c)
        batch_time.update(time.time() - end)
        losses.update(loss.data[0], images.size(0))
        end = time.time()
        # im2show = np.copy(np.array(trans(images[0])))
        raw = cv2.imread(image_paths[0])
        im2show = np.copy(raw)
        bbox = final.data[0].cpu().numpy()
        bbox = [int(x) for x in bbox]
        cv2.rectangle(im2show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 4)
        # width = bbox[2] - bbox[0]
        # height = bbox[3] - bbox[1]
        #
        # cv2.rectangle(im2show, (bbox[4]-width//2, bbox[5]-height//2), (bbox[4]+width//2, bbox[5]+height//2), (0, 255, 255), 4)
        # cv2.rectangle(im2show, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + 20, bbox[1] + 20), (255, 255, 0), 4)
        cv2.imshow('test', im2show)
        cv2.waitKey(2000)
        if num_iter % display == 0:
            print_info = 'Eval: [{iter}/{tot}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.avg:.3f}\t' 'Loss {loss.avg:.4f}\t' \
                         ''.format(iter = num_iter, tot = len(val_data_loader), batch_time=batch_time,
                                   data_time=data_time, loss=losses)
            print(print_info)
            logger.append(print_info)
    return logger


def cls_predict(val_data_loader, model, criterion, display):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    result = np.array([], dtype=int)
    ious = np.array([], dtype=float)
    images_list = []
    end = time.time()
    logger = []
    trans = ToPILImage()
    for num_iter, (images, _, image_paths, bboxs, bboxs_c, params) in enumerate(val_data_loader):
        for image_file in image_paths:
            images_list.append(image_file)
        data_time.update(time.time() - end)
        final, map = model(Variable(images))
        # loss = criterion(final, bbox_od)
        batch_time.update(time.time() - end)
        end = time.time()
        # im2show = np.copy(np.array(trans(images[0])))
        # raw = cv2.imread(image_paths[0])
        pred_od_bboxs = final.data.cpu().numpy()
        gt_od_bboxs = bboxs.numpy()
        tmp, tmp_ious = get_detect_od_array(pred_od_bboxs, gt_od_bboxs)
        result = np.append(result, tmp)
        ious = np.append(ious, tmp_ious)

        raw = Image.open(image_paths[0])
        raw,_,_,_ = scale_image(raw, 512)
        raw = np.array(raw)
        im2show = np.copy(raw)
        im2show = cv2.cvtColor(im2show, cv2.COLOR_RGB2BGR)
        bbox = final.data[0].cpu().numpy()
        bbox = [int(x) for x in bbox]
        param = [params[0][0], params[1][0], params[2][0]]
        # bbox = pts_trans_inv(bbox, param[0], param[1], param[2])
        cv2.rectangle(im2show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 4)
        # width = bbox[2] - bbox[0]
        # height = bbox[3] - bbox[1]
        #
        # cv2.rectangle(im2show, (bbox[4]-width//2, bbox[5]-height//2), (bbox[4]+width//2, bbox[5]+height//2), (0, 255, 255), 4)
        cv2.rectangle(im2show, (bbox[4] - 20, bbox[5] - 20), (bbox[4] + 20, bbox[5] + 20), (255, 255, 0), 4)
        cv2.circle(im2show, (bbox[4], bbox[5]), 4, (0, 255, 255))
        cv2.imshow('test', im2show)
        cv2.waitKey(2000)

    print_info = '[optic_disc detection]:\tthreshold:{}\tdetection accuracy:{}'.format(0.5, result.sum() / len(result))
    print(print_info)
    logger.append(print_info)

    assert len(ious) == len(images_list)
    error_image_list = []
    error_thres = 0.5
    for i in range(len(ious)):
        if ious[i] < error_thres:
            error_image_list.append(images_list[i])
    print(error_image_list)
    logger.append(''.format(error_image_list))

    return logger


def main():
    print('===> Parsing options')
    opt = parse_args()
    print(opt)
    cudnn.benchmark = True
    torch.manual_seed(opt.seed)
    if not os.path.isdir(opt.output):
        os.makedirs(opt.output)
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    output_dir = os.path.join(opt.output,
                              opt.dataset + '_od_and_fovea_detection_' + opt.phase + '_' + time_stamp + '_' + opt.model + '_' + opt.exp)
    if not os.path.exists(output_dir):
        print('====> Creating ', output_dir)
        os.makedirs(output_dir)
    print('====> Building model:')
    # model = cls_model(opt.model, opt.crop, 2, opt.weight)
    model = cls_model_od_and_fovea(opt.model, opt.crop, 2, opt.weight)
    criterion = torch.nn.MSELoss()
    best_loss = 1e4
    if opt.phase == 'train':
        print('====> Training model:')
        dataloader_train = torch.utils.data.DataLoader(
            DRDetectionDS_od_and_fovea_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/incurable',
                              '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/incurable',
                              448, 512),
            shuffle=True, pin_memory=True, batch_size=10)
        dataloader_val = torch.utils.data.DataLoader(
            DRDetectionDS_od_and_fovea_predict_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
                              '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
                              512),
            shuffle=True, pin_memory=True, batch_size=10)
        for epoch in range(opt.epoch):
            if epoch < opt.fix:
                lr = opt.lr
            else:
                lr = opt.lr * (0.1 ** (epoch // opt.step))
            optimizer = optim.SGD([{'params': model.base.parameters()}, {'params': model.cls.parameters()}], lr=opt.lr,
                                  momentum=opt.mom, weight_decay=opt.wd, nesterov=True)
            logger = cls_train(dataloader_train, nn.DataParallel(model).cuda(), criterion, optimizer, epoch, opt.display)
            # if epoch % 100 == 0:
            #     torch.save(model.cpu().state_dict(), os.path.join(output_dir,
            #                                                       opt.dataset + '_od_and_fovea_detection_' + opt.model + '_%05d' % epoch + '_best.pth'))
            #     print('====> Save model: {}'.format(
            #         os.path.join(output_dir,
            #                      opt.dataset + '_od_and_fovea_detection_' + opt.model + '_%05d' % epoch + '_best.pth')))
            # if not os.path.isfile(os.path.join(output_dir, 'train.log')):
            #     with open(os.path.join(output_dir, 'train.log'), 'w') as fp:
            #         fp.write(str(opt)+'\n\n')
            # with open(os.path.join(output_dir, 'train.log'), 'a') as fp:
            #     fp.write('\n' + '\n'.join(logger))

            logger_val, loss_val = cls_val(dataloader_val, nn.DataParallel(model).cuda(), criterion, opt.display)
            if loss_val < best_loss:
                best_loss = loss_val
                print('====> Current best validation loss is: {}'.format(best_loss))
                torch.save(model.cpu().state_dict(), os.path.join(output_dir,
                                                              opt.dataset + '_od_and_fovea_detection_' + opt.model + '_%05d' % epoch + '_best.pth'))
                print('====> Save model: {}'.format(os.path.join(output_dir, opt.dataset + '_od_and_fovea_detection_' + opt.model + '_%05d' % epoch + '_best.pth')))
            if not os.path.isfile(os.path.join(output_dir, 'train.log')):
                with open(os.path.join(output_dir, 'train.log'), 'w') as fp:
                    fp.write(str(opt)+'\n\n')
            with open(os.path.join(output_dir, 'train.log'), 'a') as fp:
                fp.write('\n' + '\n'.join(logger))
                fp.write('\n' + '\n'.join(logger_val))
    elif opt.phase == 'test':
        if opt.weight:
            print('====> Evaluating model')
            # dataloader = torch.utils.data.DataLoader(
            #     DRDetectionDS_predict_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
            #                       '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
            #                       512),
            #     shuffle=False, pin_memory=False, batch_size=1)
            # logger = cls_eval(dataloader, nn.DataParallel(model).cuda(), criterion, opt.display)
            root = '//home/weidong/code/github/ex'
            ds = DRDetection_predict_raw(root, root, 512)
            dataloader = DataLoader(ds, batch_size=2, shuffle=False)
            logger = cls_predict(dataloader, nn.DataParallel(model).cuda(), criterion, opt.display)
            if not os.path.isfile(os.path.join(output_dir, 'predict.log')):
                with open(os.path.join(output_dir, 'predict.log'), 'w') as fp:
                    fp.write(str(opt)+'\n\n')
            with open(os.path.join(output_dir, 'predict.log'), 'a') as fp:
                fp.write('\n' + '\n'.join(logger))

if __name__ == '__main__':
    main()