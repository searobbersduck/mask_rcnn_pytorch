import sys

sys.path.append('../')

from models.faster_rcnn import create_model


import config

import argparse
import time
import glob
import os

import torch
from paper_data.data import DRDetectionDS, coco_collate
# from paper_data.data_xml import DRDetectionDS_xml, coco_collate

import numpy as np

from utils import (save_checkpoint, AverageMeter, adjust_learning_rate,
                   get_optimizer)

import pandas as pd

import cv2

model_names = list(map(lambda n: os.path.basename(n)[:-3],
                       glob.glob('models/[A-Za-z]*.py')))

parser = argparse.ArgumentParser(
                description='Image classification PK main script')

exp_group = parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/default-CLOCKTIME)')
exp_group.add_argument('--resume', default='', type=str, metavar='PATH',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--evaluate', dest='evaluate', default='',
                       choices=['', 'val', 'test'],
                       help='eval mode: evaluate model on val/test set'
                       ' (default: training mode)')
exp_group.add_argument('-f', '--force', dest='force', action='store_true',
                       help='force to overwrite existing save path')
exp_group.add_argument('--print-freq', '-p', default=2, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--no_tensorboard', dest='tensorboard',
                       action='store_false',
                       help='do not use tensorboard_logger for logging')

# dataset related
data_group = parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='coco-debug',
                        choices=config.datasets.keys(),
                        help='datasets: ' +
                        ' | '.join(config.datasets.keys()) +
                        ' (default: coco-train-minival)')
data_group.add_argument('--data-root', metavar='DIR', default='data/COCO',
                        help='path to dataset (default: data)')
data_group.add_argument('-j', '--num-workers', dest='num_workers', default=4,
                        type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
data_group.add_argument('--normalized', action='store_true',
                        help='normalize the data into zero mean and unit std')

# model arch related
arch_group = parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='faster_rcnn',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: faster_rcnn)')
arch_group.add_argument('--backbone', metavar='BACKBONE', default='resnet-50-c4',
                        type=str, help='backbone of RPN')
# arch_group.add_argument('-d', '--depth', default=56, type=int, metavar='D',
#                         help='depth (default=56)')
# arch_group.add_argument('--drop-rate', default=0.0, type=float,
#                         metavar='DROPRATE', help='dropout rate (default: 0.2)')
# arch_group.add_argument('--death-mode', default='none',
#                         choices=['none', 'linear', 'uniform'],
#                         help='death mode (default: none)')
# arch_group.add_argument('--death-rate', default=0.5, type=float,
#                         help='death rate rate (default: 0.5)')
# arch_group.add_argument('--bn-size', default=4, type=int,
#                         metavar='B', help='bottle neck ratio for DenseNet'
#                         ' (0 means dot\'t use bottle necks) (default: 4)')
# arch_group.add_argument('--compression', default=0.5, type=float,
#                         metavar='C', help='compression ratio for DenseNet'
#                         ' (1 means dot\'t use compression) (default: 0.5)')
# used to set the argument when to resume automatically
# arch_resume_names = ['arch', 'depth', 'death_mode', 'death_rate', 'death_rate',
                     # 'bn_size', 'compression']

# training related
optim_group = parser.add_argument_group('optimization', 'optimization setting')
optim_group.add_argument('--niters', default=160000, type=int, metavar='N',
                         help='number of total iterations to run (default: 160000)')
optim_group.add_argument('--start-iter', default=1, type=int, metavar='N',
                         help='manual iter number (useful on restarts, default: 1)')
optim_group.add_argument('--eval-freq', default=1000, type=int, metavar='N',
                         help='number of iterations to run before evaluation (default: 1000)')
optim_group.add_argument('--patience', default=0, type=int, metavar='N',
                         help='patience for early stopping'
                         '(0 means no early stopping)')
optim_group.add_argument('-b', '--batch-size', default=16, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
optim_group.add_argument('--optimizer', default='sgd',
                         choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                         help='optimizer (default=sgd)')
optim_group.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                         metavar='LR',
                         help='initial learning rate (default: 0.02)')
optim_group.add_argument('--decay_rate', default=0.1, type=float, metavar='N',
                         help='decay rate of learning rate (default: 0.1)')
optim_group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                         help='momentum (default=0.9)')
optim_group.add_argument('--no_nesterov', dest='nesterov',
                         action='store_false',
                         help='do not use Nesterov momentum')
optim_group.add_argument('--alpha', default=0.001, type=float, metavar='M',
                         help='alpha for Adam (default: 0.001)')
optim_group.add_argument('--beta1', default=0.9, type=float, metavar='M',
                         help='beta1 for Adam (default: 0.9)')
optim_group.add_argument('--beta2', default=0.999, type=float, metavar='M',
                         help='beta2 for Adam (default: 0.999)')
optim_group.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                         metavar='W', help='weight decay (default: 1e-4)')


args = parser.parse_args()

args.config_of_data = config.datasets[args.data]

dict = {}

model = create_model(None, None, 3, backbone='resnet-50-c4', **dict)

model.load_state_dict(torch.load('./output/detector_3.pth'))

# model.cuda()

print(args)

print('pause')


train_loader = torch.utils.data.DataLoader(DRDetectionDS('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/dr_ann',
                                                           '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/detection_xml1.txt',
                                                         512.0001), batch_size=16, collate_fn=coco_collate,
                                             shuffle=True, pin_memory=True)

val_loader = torch.utils.data.DataLoader(DRDetectionDS('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/dr_ann',
                                                           '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/detection_xml1.txt',
                                                           512.0001), batch_size=1, collate_fn=coco_collate,
                                             shuffle=False, pin_memory=False)



# train_loader = torch.utils.data.DataLoader(DRDetectionDS('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
#                                                            '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/detection_xml.txt',
#                                                          512.0001), batch_size=16, collate_fn=coco_collate,
#                                              shuffle=True, pin_memory=True)
#
# val_loader = torch.utils.data.DataLoader(DRDetectionDS('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
#                                                            '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/detection_xml.txt',
#                                                            512.0001), batch_size=1, collate_fn=coco_collate,
#                                              shuffle=False, pin_memory=False)

# train_loader = torch.utils.data.DataLoader(DRDetectionDS_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
#                                                            '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
#                                                            1000), batch_size=2, collate_fn=coco_collate,
#                                              shuffle=True, pin_memory=True)
#
# val_loader = torch.utils.data.DataLoader(DRDetectionDS_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
#                                                            '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
#                                                            1000), batch_size=1, collate_fn=coco_collate,
#                                              shuffle=False, pin_memory=False)

def train(train_loader, model, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter()
    rpn_losses = AverageMeter()
    odn_losses = AverageMeter()
    rpn_ce_losses = AverageMeter()
    rpn_box_losses = AverageMeter()
    odn_ce_losses = AverageMeter()
    odn_box_losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (images, anns, image_paths) in enumerate(train_loader):
        lr = 0.001
        optimizer.zero_grad()
        for j, image in enumerate(images):
            input_anns = anns[j]
            if len(input_anns) == 0:
                continue
            gt_bbox = np.vstack([ann['bbox']+[ann['ordered_id']] for ann in input_anns])
            im_info = [[image.size(1), image.size(2), input_anns[0]['scale_ratio']]]
            input_var = torch.autograd.Variable(image.unsqueeze(0).cuda(),
                                                requires_grad=False)

            cls_prob, bbox_pred, rois = model(input_var, im_info, gt_bbox)
            loss = model.loss
            loss.backward()

            total_losses.update(loss.data[0], input_var.size(0))
            rpn_losses.update(model.rpn.loss.data[0], input_var.size(0))
            rpn_ce_losses.update(
                model.rpn.cross_entropy.data[0], input_var.size(0))
            rpn_box_losses.update(
                model.rpn.loss_box.data[0], input_var.size(0))
            odn_losses.update(model.odn.loss.data[0], input_var.size(0))
            odn_ce_losses.update(
                model.odn.cross_entropy.data[0], input_var.size(0))
            odn_box_losses.update(
                model.odn.loss_box.data[0], input_var.size(0))

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_freq > 0 and (i + 1) % args.print_freq == 0:
            print('iter: [{0}] '
                  'Time {batch_time.val:.3f} '
                  'Data {data_time.val:.3f} '
                  'Loss {total_losses.val:.4f} '
                  'RPN {rpn_losses.val:.4f} '
                  '{rpn_ce_losses.val:.4f} '
                  '{rpn_box_losses.val:.4f} '
                  'ODN {odn_losses.val:.4f} '
                  '{odn_ce_losses.val:.4f} '
                  '{odn_box_losses.val:.4f} '
                  .format(i, batch_time=batch_time,
                          data_time=data_time,
                          total_losses=total_losses,
                          rpn_losses=rpn_losses,
                          rpn_ce_losses=rpn_ce_losses,
                          rpn_box_losses=rpn_box_losses,
                          odn_losses=odn_losses,
                          odn_ce_losses=odn_ce_losses,
                          odn_box_losses=odn_box_losses))

optimizer = get_optimizer(model, args)

def validate(val_loader, model):
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    image_list = []
    detected_info_list = []
    for i, (inputs, anns, image_paths) in enumerate(val_loader):
        for img in image_paths:
            image_list.append(img)
        for j, input in enumerate(inputs):
            input_anns = anns[j]
            im_info = [[input.size(1), input.size(2), input_anns[0]['scale_ratio']]]
            input_var = torch.autograd.Variable(input.unsqueeze(0), requires_grad = False).cuda()
            cls_prob, bbox_pred, rois = model(input_var, im_info)
            scores, pred_boxes, classes = model.interpret_outputs(cls_prob, bbox_pred, rois, im_info)
            detected_info_list.append('{}|{}'.format(scores, pred_boxes))
            print(scores, pred_boxes)
        batch_time.update(time.time()-end)
        end = time.time()

    assert len(image_list) == len(detected_info_list)

import torchvision.transforms as transforms
from PIL import Image

def inference(img, model):
    model.eval()
    image_cv = cv2.imread(img)
    im2show = np.copy(image_cv)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = trans(Image.open(img))
    input_var = torch.autograd.Variable(image.unsqueeze(0)).cuda()
    im_info = [[image.size(1), image.size(2), 1]]
    cls_prob, bbox_pred, rois = model(input_var, im_info)
    scores, pred_boxes, classes = model.interpret_outputs(cls_prob, bbox_pred, rois, im_info, min_score=0.9)
    if len(scores) > 0:
        print('{}'.format(os.path.basename('img')), scores, pred_boxes, classes)
        for i,det in enumerate(pred_boxes):
            det = tuple(int(x) for x in det)
            cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
            cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
        cv2.imshow('demo', im2show)
        cv2.waitKey(0)


    # im2show = np.copy(image)
    # for i, det in enumerate(dets):
    #     det = tuple(int(x) for x in det)
    #     cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
    #     cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
    #                 1.0, (0, 0, 255), thickness=1)


output_root = './output'
os.makedirs(output_root, exist_ok=True)

def adjust_learning_rate1(optimizer, lr_init, decay_rate, epoch, num_epochs):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch >= num_epochs * 0.75:
        lr *= decay_rate**2
    elif epoch >= num_epochs * 0.5:
        lr *= decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# for i in range(100000):
#     print('===> Loop: {}'.format(i))
#     lr = adjust_learning_rate1(optimizer, args.lr, args.decay_rate,
#                               i, 10000)
#     train_loss = train(train_loader, model.cuda(), optimizer)
#     if i % 100 == 0:
#         torch.save(model.cpu().state_dict(), os.path.join(output_root, 'detector_{}.pth'.format(i//100)))
#         print('====> save model in {}'.format(os.path.join(output_root, 'detector_{}.pth'.format(i//100))))

# validate(val_loader, model.cuda())




# inference
from glob import glob
infer_img_list = glob('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/dr_ann/*.png')
for index in infer_img_list:
    inference(index, model.cuda())

