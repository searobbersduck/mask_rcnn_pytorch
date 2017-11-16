import os
import sys
sys.path.append('../')

import torch.nn as nn
import torchvision
import torch

import math

from utils import AverageMeter
import time
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, Scale

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

model_type = ['od', 'fovea', 'od_fovea']

class cls_model_c(nn.Module):
    def __init__(self, name, inmap, type, weights=None, scratch=False):
        if type == model_type[0]:
            num = 2
        elif type == model_type[1]:
            num = 1
        else:
            num = 3
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
        cls = nn.Sequential(nn.AvgPool2d(featmap), nn.Conv2d(planes, num*2, kernel_size=1, stride=1, padding=0, bias=True))
        initial_cls_weights(cls)
        self.cls = cls
        if weights:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        map = self.base(x)
        y = self.cls(map)
        y = y.view(y.size(0), -1)
        return y, map

class cls_model_od(cls_model_c):
    def __init__(self, name, inmap, bbox_num, weights=None, scratch=False):
        super(cls_model_od, self).__init__(name, inmap, model_type[0], weights, scratch)


class cls_model_fovea(cls_model_c):
    def __init__(self, name, inmap, bbox_num, weights=None, scratch=False):
        super(cls_model_fovea, self).__init__(name, inmap, model_type[1], weights, scratch)

class cls_model_od_and_fovea(cls_model_c):
    def __init__(self, name, inmap, bbox_num, weights=None, scratch=False):
        super(cls_model_od_and_fovea, self).__init__(name, inmap, model_type[2], weights, scratch)


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


