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

from predict_common import DRDetectionDS_predict
from predict_common import DRDetection_predict_raw, \
    pts_trans_inv, scale_image, get_detect_od_array, get_detect_fovea_array

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

class DRDetectionDS_xml(Dataset):
    def __init__(self, root, ann_root, crop, size, scale_size=None):
        super(DRDetectionDS_xml, self).__init__()
        self.root = root
        self.ann_root = ann_root
        self.scale_size = scale_size
        self.crop = crop
        self.size = size
        self.ratio = size/crop
        self.padding = crop//2
        self.transform = transforms.Compose([
            Scale(self.size),
            PILColorJitter(),
            transforms.ToTensor(),
            # Lighting(alphastd=0.01, eigval=eigen_values, eigvec=eigen_values),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.classid = ['optic_dis', 'macular']
        ann_list = glob(os.path.join(ann_root, '*.xml'))
        img_list = glob(os.path.join(root, '*.png'))
        img_list = [i.split('.')[0] for i in img_list]
        ann_list = [i.split('.')[0] for i in ann_list]
        self.data_list = []
        self.ann_info_list = []
        self.bboxs_list = []
        self.bboxs_c_list = []
        for ann in ann_list:
            if ann in img_list:
                self.data_list.append(ann)
        for index in self.data_list:
            anns, bbox, bbox_c = self.__read_xml(index)
            self.ann_info_list.append(anns)
            self.bboxs_list.append(bbox)
            self.bboxs_c_list.append(bbox_c)


    def __read_xml(self, xml_file):
        xml_file = os.path.join(self.ann_root, xml_file+'.xml')
        tree = ET.parse(xml_file)
        anns = []
        pts = np.array([], dtype=int)
        pts_c = np.array([], dtype=int)
        for obj in tree.getiterator('object'):
            if (obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc' or obj.find('name').text == 'optic-disc'):
                ann = {}
                # ann['cls_id'] = obj.find('name').text
                ann['ordered_id'] = 1 if (obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc') else 2
                # ann['bbox'] = [0] * 4
                xmin = obj.find('bndbox').find('xmin')
                ymin = obj.find('bndbox').find('ymin')
                xmax = obj.find('bndbox').find('xmax')
                ymax = obj.find('bndbox').find('ymax')
                ann['bbox'] = np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
                ann['scale_ratio'] = 1
                ann['area'] = (int(ymax.text)-int(ymin.text)) * (int(xmax.text)-int(xmin.text))
                anns.append(ann)
                pts = np.append(pts, np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)]))
                pts_c = np.append(pts_c, np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)]))
        for obj in tree.getiterator('object'):
            if obj.find('name').text == 'macular':
                ann = {}
                # ann['cls_id'] = obj.find('name').text
                ann['ordered_id'] = 1 if (obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc' or obj.find('name').text == 'optic-disc') else 2
                # ann['bbox'] = [0] * 4
                xmin = obj.find('bndbox').find('xmin')
                ymin = obj.find('bndbox').find('ymin')
                xmax = obj.find('bndbox').find('xmax')
                ymax = obj.find('bndbox').find('ymax')
                ann['bbox'] = np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
                ann['scale_ratio'] = 1
                ann['area'] = (int(ymax.text)-int(ymin.text)) * (int(xmax.text)-int(xmin.text))
                anns.append(ann)
                pts = np.append(pts, np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)]))
                pts_c = np.append(pts_c, np.array([(int(xmin.text)+int(xmax.text))//2, (int(ymin.text)+int(ymax.text))//2]))
        return anns,pts, pts_c[4:6]

    def __clamp(self, min_val, max_val, val):
        return max(min_val, min(max_val, val))

    def __bbox_trans(self, bboxs, l, u, ratio):
        assert len(bboxs) == 8
        pts = np.array([], dtype=int)
        for i in range(len(bboxs)):
            if i % 2 == 0:
                pts = np.append(pts, self.__clamp(0, self.size-1, int((bboxs[i] - l) * ratio)))
            else:
                pts = np.append(pts, self.__clamp(0, self.size-1, int((bboxs[i] - u) * ratio)))
        return pts

    def __bbox_trans_center(self, bboxs, l, u, ratio):
        assert len(bboxs) == 2
        pts = np.array([], dtype=int)
        for i in range(len(bboxs)):
            if i % 2 == 0:
                pts = np.append(pts, self.__clamp(0, self.size-1, int((bboxs[i] - l) * ratio)))
            else:
                pts = np.append(pts, self.__clamp(0, self.size-1, int((bboxs[i] - u) * ratio)))
        return pts

    def __get_random_bbox(self, padding, size):
        x_center = random.randint(0 + padding, size - padding)
        y_center = random.randint(0 + padding, size - padding)
        return x_center - padding, y_center - padding, x_center + padding, y_center + padding


    def __getitem__(self, item):
        anns = self.ann_info_list[item]
        image_path = os.path.join(self.root, self.data_list[item]+'.png')
        img = Image.open(image_path)
        l,u,r,b = self.__get_random_bbox(self.padding, self.size)
        img = img.crop((l,u,r,b))

        if self.scale_size is not None:
            w,h = img.size
            scale_ratio = self.scale_size/w if w<h else self.scale_size/h
            if scale_ratio != 1:
                img = img.resize((int(w*scale_ratio), int(h*scale_ratio)), Image.BILINEAR)
                for ann in anns:
                    ann['area'] *= scale_ratio**2
                    ann['bbox'] = [x*scale_ratio for x in ann['bbox']]
                    ann['scale_ratio'] = scale_ratio

        img = self.transform(img)

        return img, anns, image_path, self.__bbox_trans(self.bboxs_list[item], l, u, self.ratio), \
               self.__bbox_trans_center(self.bboxs_c_list[item], l, u, self.ratio)

    def __len__(self):
        return len(self.data_list)

class DRDetectionDS_predict_xml(Dataset):
    def __init__(self, root, ann_root, scale_size=None):
        super(DRDetectionDS_predict_xml, self).__init__()
        self.root = root
        self.ann_root = ann_root
        self.scale_size = scale_size
        self.transform = transforms.Compose([
            PILColorJitter(),
            transforms.ToTensor(),
            # Lighting(alphastd=0.01, eigval=eigen_values, eigvec=eigen_values),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.classid = ['optic_dis', 'macular']
        ann_list = glob(os.path.join(ann_root, '*.xml'))
        img_list = glob(os.path.join(root, '*.png'))
        img_list = [i.split('.')[0] for i in img_list]
        ann_list = [i.split('.')[0] for i in ann_list]
        self.data_list = []
        self.ann_info_list = []
        self.bboxs_list = []
        self.bboxs_c_list = []
        for ann in ann_list:
            if ann in img_list:
                self.data_list.append(ann)
        for index in self.data_list:
            anns, bbox, bbox_c = self.__read_xml(index)
            self.ann_info_list.append(anns)
            self.bboxs_list.append(bbox)
            self.bboxs_c_list.append(bbox_c)


    def __read_xml(self, xml_file):
        xml_file = os.path.join(self.ann_root, xml_file+'.xml')
        tree = ET.parse(xml_file)
        anns = []
        pts = np.array([], dtype=int)
        pts_c = np.array([], dtype=int)
        for obj in tree.getiterator('object'):
            if (obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc' or obj.find('name').text == 'optic-disc'):
                ann = {}
                # ann['cls_id'] = obj.find('name').text
                ann['ordered_id'] = 1 if (obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc') else 2
                # ann['bbox'] = [0] * 4
                xmin = obj.find('bndbox').find('xmin')
                ymin = obj.find('bndbox').find('ymin')
                xmax = obj.find('bndbox').find('xmax')
                ymax = obj.find('bndbox').find('ymax')
                ann['bbox'] = np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
                ann['scale_ratio'] = 1
                ann['area'] = (int(ymax.text)-int(ymin.text)) * (int(xmax.text)-int(xmin.text))
                anns.append(ann)
                pts = np.append(pts, np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)]))
                pts_c = np.append(pts_c, np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)]))
        for obj in tree.getiterator('object'):
            if obj.find('name').text == 'macular':
                ann = {}
                # ann['cls_id'] = obj.find('name').text
                ann['ordered_id'] = 1 if (obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc' or obj.find('name').text == 'optic-disc') else 2
                # ann['bbox'] = [0] * 4
                xmin = obj.find('bndbox').find('xmin')
                ymin = obj.find('bndbox').find('ymin')
                xmax = obj.find('bndbox').find('xmax')
                ymax = obj.find('bndbox').find('ymax')
                ann['bbox'] = np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
                ann['scale_ratio'] = 1
                ann['area'] = (int(ymax.text)-int(ymin.text)) * (int(xmax.text)-int(xmin.text))
                anns.append(ann)
                pts = np.append(pts, np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)]))
                pts_c = np.append(pts_c, np.array([(int(xmin.text)+int(xmax.text))//2, (int(ymin.text)+int(ymax.text))//2]))
        return anns,pts, pts_c[4:6]

    def __getitem__(self, item):
        anns = self.ann_info_list[item]
        image_path = os.path.join(self.root, self.data_list[item]+'.png')
        img = Image.open(image_path)

        if self.scale_size is not None:
            w,h = img.size
            scale_ratio = self.scale_size/w if w<h else self.scale_size/h
            if scale_ratio != 1:
                img = img.resize((int(w*scale_ratio), int(h*scale_ratio)), Image.BILINEAR)
                for ann in anns:
                    ann['area'] *= scale_ratio**2
                    ann['bbox'] = [x*scale_ratio for x in ann['bbox']]
                    ann['scale_ratio'] = scale_ratio

        img = self.transform(img)

        return img, anns, image_path, self.bboxs_list[item], self.bboxs_c_list[item]

    def __len__(self):
        return len(self.data_list)


def coco_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size, or put collade recursively for dict"
    if isinstance(batch[0], tuple):
        # if each batch element is not a tensor, then it should be a tuple
        # of tensors; in that case we collate each element in the tuple
        transposed = zip(*batch)
        return [coco_collate(samples) for samples in transposed]
    return batch

def initial_cls_weights(cls):
    for m in cls.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            m.weight.data.normal_(0, math.sqrt(2./n))
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

class cls_model(nn.Module):
    def __init__(self, name, inmap, bbox_num, weights=None, scratch=False):
        super(cls_model, self).__init__()
        if name == 'rsn152':
            base_model = torchvision.models.resnet152()
            featmap = inmap // 32
            planes = 2048
        elif name == 'rsn101':
            base_model = torchvision.models.resnet101()
            featmap = inmap // 32
            planes = 2048
        elif name == 'rsn50':
            base_model = torchvision.models.resnet50()
            featmap = inmap // 32
            planes = 2048
        elif name == 'rsn34':
            base_model = torchvision.models.resnet34()
            featmap = inmap // 32
            planes = 512
        elif name == 'rsn18':
            base_model = torchvision.models.resnet18()
            featmap = inmap // 32
            planes = 512
        if not scratch:
            base_model.load_state_dict(torch.load('../pretrained/'+name+'.pth'))
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        cls = nn.Sequential(nn.AvgPool2d(featmap), nn.Conv2d(planes, bbox_num*2*2, kernel_size=1, stride=1, padding=0, bias=True))
        initial_cls_weights(cls)
        self.cls = cls
        if weights:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        map = self.base(x)
        y = self.cls(map)
        y = y.view(y.size(0), -1)
        return y, map

class cls_model_c(nn.Module):
    def __init__(self, name, inmap, bbox_num, weights=None, scratch=False):
        super(cls_model_c, self).__init__()
        if name == 'rsn152':
            base_model = torchvision.models.resnet152()
            featmap = inmap // 32
            planes = 2048
        elif name == 'rsn101':
            base_model = torchvision.models.resnet101()
            featmap = inmap // 32
            planes = 2048
        elif name == 'rsn50':
            base_model = torchvision.models.resnet50()
            featmap = inmap // 32
            planes = 2048
        elif name == 'rsn34':
            base_model = torchvision.models.resnet34()
            featmap = inmap // 32
            planes = 512
        elif name == 'rsn18':
            base_model = torchvision.models.resnet18()
            featmap = inmap // 32
            planes = 512
        if not scratch:
            base_model.load_state_dict(torch.load('../pretrained/'+name+'.pth'))
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        cls = nn.Sequential(nn.AvgPool2d(featmap), nn.Conv2d(planes, 1*2, kernel_size=1, stride=1, padding=0, bias=True))
        initial_cls_weights(cls)
        self.cls = cls
        if weights:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        map = self.base(x)
        y = self.cls(map)
        y = y.view(y.size(0), -1)
        return y, map


def cls_train(train_data_loader, model, criterion, optimizer, epoch, display):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (images, anns, image_paths, bboxs, bboxs_c) in enumerate(train_data_loader):
        data_time.update(time.time() - end)
        # bbox_od = Variable(anns[1]['bbox'].type(torch.FloatTensor).cuda())
        # bbox_od = Variable(bboxs.type(torch.FloatTensor).cuda())
        bbox_od_c = Variable(bboxs_c.type(torch.FloatTensor).cuda())
        final, map = model(Variable(images))
        # loss = criterion(final, bbox_od)
        loss = criterion(final, bbox_od_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        losses.update(loss.data[0], images.size(0))
        end = time.time()
        if num_iter % display == 0:
            print_info = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.avg:.3f}\t' 'Loss {loss.avg:.4f}\t' \
                         ''.format(epoch, num_iter, len(train_data_loader), batch_time=batch_time,
                                   data_time=data_time, loss=losses)
            print(print_info)
            logger.append(print_info)
    return logger

def cls_val(val_data_loader, model, criterion, display):
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
        if num_iter % display == 0:
            print_info = 'Eval: [{iter}/{tot}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.avg:.3f}\t' 'Loss {loss.avg:.4f}\t' \
                         ''.format(iter = num_iter, tot = len(val_data_loader), batch_time=batch_time,
                                   data_time=data_time, loss=losses)
            print(print_info)
            logger.append(print_info)
    return logger, losses.avg

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
        # cv2.rectangle(im2show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 4)
        # width = bbox[2] - bbox[0]
        # height = bbox[3] - bbox[1]
        #
        # cv2.rectangle(im2show, (bbox[4]-width//2, bbox[5]-height//2), (bbox[4]+width//2, bbox[5]+height//2), (0, 255, 255), 4)
        cv2.rectangle(im2show, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + 20, bbox[1] + 20), (255, 255, 0), 4)
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
    end = time.time()
    logger = []
    trans = ToPILImage()
    for num_iter, (images, _, image_paths, _, _) in enumerate(val_data_loader):
        data_time.update(time.time() - end)
        final, map = model(Variable(images))
        # loss = criterion(final, bbox_od)
        batch_time.update(time.time() - end)
        end = time.time()
        # im2show = np.copy(np.array(trans(images[0])))
        raw = cv2.imread(image_paths[0])
        im2show = np.copy(raw)
        bbox = final.data[0].cpu().numpy()
        bbox = [int(x) for x in bbox]
        # cv2.rectangle(im2show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 4)
        # width = bbox[2] - bbox[0]
        # height = bbox[3] - bbox[1]
        #
        # cv2.rectangle(im2show, (bbox[4]-width//2, bbox[5]-height//2), (bbox[4]+width//2, bbox[5]+height//2), (0, 255, 255), 4)
        cv2.rectangle(im2show, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + 20, bbox[1] + 20), (255, 255, 0), 4)
        cv2.imshow('test', im2show)
        cv2.waitKey(2000)
    return logger


def cls_predict_raw_ann(val_data_loader, model, criterion, display):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    result = np.array([], dtype=int)
    bias = np.array([], dtype=float)
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

        pred_fovea_center = final.data.cpu().numpy()
        gt_od_bboxs = bboxs.numpy()
        gt_fovea_center = bboxs_c.numpy()[:,4:6]
        tmp, tmp_bias = get_detect_fovea_array(pred_fovea_center, gt_fovea_center, gt_od_bboxs)
        result = np.append(result, tmp)
        bias = np.append(bias, tmp_bias)

        # im2show = np.copy(np.array(trans(images[0])))
        # raw = cv2.imread(image_paths[0])
        # im2show = np.copy(raw)
        # bbox = final.data[0].cpu().numpy()
        # bbox = [int(x) for x in bbox]
        # # cv2.rectangle(im2show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 4)
        # # width = bbox[2] - bbox[0]
        # # height = bbox[3] - bbox[1]
        # #
        # # cv2.rectangle(im2show, (bbox[4]-width//2, bbox[5]-height//2), (bbox[4]+width//2, bbox[5]+height//2), (0, 255, 255), 4)
        # cv2.rectangle(im2show, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + 20, bbox[1] + 20), (255, 255, 0), 4)
        # cv2.imshow('test', im2show)
        # cv2.waitKey(2000)
    print('threshold:{}\tdetection accuracy:{}'.format(0.5, result.sum() / len(result)))
    assert len(bias) == len(images_list)
    error_image_list = []
    error_thres = 1.0
    for i in range(len(bias)):
        if bias[i] > error_thres:
            error_image_list.append(images_list[i])
    print(error_image_list)

    return logger


# dataloader = torch.utils.data.DataLoader(DRDetectionDS_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
#                                                            '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
#                                                            512),
#                                              shuffle=True, pin_memory=True, batch_size=10)
#
# model = cls_model('rsn34', 512, 1, None, None)
#
# criterion = torch.nn.L1Loss()
#
# lr = 0.001
# mom = 0.9
# wd = 1e-4
# optimizer = optim.SGD([{'params': model.base.parameters()}, {'params': model.cls.parameters()}], lr=lr, momentum=mom, weight_decay=wd, nesterov=True)
#
#
#
#
# for epoch in range(10000):
#     cls_train(dataloader, nn.DataParallel(model).cuda(), criterion, optimizer, epoch, 100)


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
    model = cls_model_c(opt.model, opt.crop, 2, opt.weight)
    criterion = torch.nn.MSELoss()
    best_loss = 1e4
    if opt.phase == 'train':
        print('====> Training model:')
        dataloader_train = torch.utils.data.DataLoader(
            DRDetectionDS_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/incurable',
                              '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/incurable',
                              448, 512),
            shuffle=True, pin_memory=True, batch_size=10)
        dataloader_val = torch.utils.data.DataLoader(
            DRDetectionDS_predict_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
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
            # dataloader = torch.utils.data.DataLoader(
            #     DRDetectionDS_predict_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/incurable',
            #                               '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/incurable',
            #                               512),
            #     shuffle=False, pin_memory=False, batch_size=1)





            # root = '/home/weidong/code/github/DiabeticRetinopathy_solution/data/zhizhen_new/LabelImages/512'
            # ds = DRDetectionDS_predict(root, None)
            # dataloader = DataLoader(ds, batch_size=1, shuffle=False)
            # # logger = cls_eval(dataloader, nn.DataParallel(model).cuda(), criterion, opt.display)
            # logger = cls_predict(dataloader, nn.DataParallel(model).cuda(), criterion, opt.display)

            root = '//home/weidong/code/github/ex'
            ds = DRDetection_predict_raw(root, root, 512)
            dataloader = DataLoader(ds, batch_size=2, shuffle=False)
            logger = cls_predict_raw_ann(dataloader, nn.DataParallel(model).cuda(), criterion, opt.display)

if __name__ == '__main__':
    main()