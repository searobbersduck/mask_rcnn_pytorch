'''

1. import retinal image (any size)
2. resize image to 512
3. record the resize ratio for resizing the output predicted annotation result to raw size

'''


import sys
sys.path.append('../')
from utils import PILColorJitter, AverageMeter
import torchvision.transforms as transforms
from torchvision.transforms import Scale
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import math
import random

class DRDetectionDS_predict(Dataset):
    def __init__(self, root, scale_size=None):
        super(DRDetectionDS_predict, self).__init__()
        self.root = root
        self.scale_size = scale_size
        self.transform = transforms.Compose([
            PILColorJitter(),
            transforms.ToTensor(),
            # Lighting(alphastd=0.01, eigval=eigen_values, eigvec=eigen_values),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.img_list = glob(os.path.join(root, '*.png'))
        self.data_list = []
        self.ann_info_list = []
        self.bboxs_list = []
        self.bboxs_c_list = []

    def __getitem__(self, item):
        image_path = self.img_list[item]
        img = Image.open(image_path)

        img = self.transform(img)

        return img, image_path, image_path, image_path, image_path

    def __len__(self):
        return len(self.img_list)

def test_DRDetectionDS_predict():
    root = '/home/weidong/code/github/DiabeticRetinopathy_solution/data/zhizhen_new/LabelImages/512'
    ds = DRDetectionDS_predict(root, None)
    dataloader = DataLoader(ds, batch_size=2, shuffle=False)
    for index, (imgs, _, images_path, _, _) in enumerate(dataloader):
        pil_img = transforms.ToPILImage()(imgs[0])
        pil_img.show()


def scale_image(pil_img, scale_size):
    w, h = pil_img.size
    w0, h0 = pil_img.size
    tw, th = (min(w, h), min(w, h))
    tw0, th0 = (tw, th)
    image = pil_img.crop((w // 2 - tw // 2, h // 2 - th // 2, w // 2 + tw // 2, h // 2 + th // 2))
    w, h = image.size
    tw, th = (scale_size, scale_size)
    ratio = tw / w
    assert ratio == th / h
    if ratio < 1:
        image = image.resize((tw, th), Image.ANTIALIAS)
    elif ratio > 1:
        image = image.resize((tw, th), Image.CUBIC)
    return image, w0 // 2 - tw0 // 2, h0 // 2 - th0 // 2, ratio

def pts_trans(pts, l, u, ratio):
    pts_trans = np.array([], dtype=int)
    for i in range(len(pts)):
        if i % 2 == 0:
            pts_trans = np.append(pts_trans, int((pts[i]-l)*ratio))
        else:
            pts_trans = np.append(pts_trans, int((pts[i]-u)*ratio))
    return pts_trans

def pts_trans_inv(pts, l, u, ratio):
    pts_trans = np.array([], dtype=int)
    for i in range(len(pts)):
        if i % 2 == 0:
            pts_trans = np.append(pts_trans, int((pts[i]/ratio+l)))
        else:
            pts_trans = np.append(pts_trans, int((pts[i]/ratio+u)))
    return pts_trans

def cal_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    w_i = max(0, x2-x1)
    h_i = max(0, y2-y1)
    a_i = w_i*h_i
    a_1 = int(bbox1[2]-bbox1[0])*int(bbox1[3]-bbox1[1])
    a_2 = int(bbox2[2] - bbox2[0]) * int(bbox2[3] - bbox2[1])
    a_u = a_1+a_2-a_i
    iou = a_i/a_u
    return iou

def read_xml(xml_file, ann_root):
        xml_file = os.path.join(ann_root, xml_file+'.xml')
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
        return anns,pts, pts_c[:6]


class DRDetection_predict_raw(Dataset):
    def __init__(self, root, ann_root, scale_size=None):
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
        ann_list = glob(os.path.join(ann_root, '*.xml'))
        img_list = glob(os.path.join(root, '*.jpg'))
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
            anns, bbox, bbox_c = read_xml(index, self.ann_root)
            self.ann_info_list.append(anns)
            self.bboxs_list.append(bbox)
            self.bboxs_c_list.append(bbox_c)

    def __getitem__(self, item):
        anns = self.ann_info_list[item]
        image_path = os.path.join(self.root, self.data_list[item] + '.jpg')
        print(image_path)
        img = Image.open(image_path)
        img, l, u, ratio = scale_image(img, self.scale_size)
        img = self.transform(img)
        return img, anns, image_path, \
               pts_trans(self.bboxs_list[item], l, u, ratio), \
               pts_trans(self.bboxs_c_list[item], l, u, ratio), [l,u,ratio]


    def __len__(self):
        return len(self.data_list)


def test_DRDetection_predict_raw():
    root = '//home/weidong/code/github/ex'
    ds = DRDetection_predict_raw(root, root, 512)
    dataloader = DataLoader(ds, batch_size=2, shuffle=False)
    for i, (images, anns, images_path, bboxs, bboxs_c, params) in enumerate(dataloader):
        pil_image = transforms.ToPILImage()(images[0])
        cv_image = np.array(pil_image)
        bbox = bboxs[0].numpy()
        bbox = [int(x) for x in bbox]
        bbox_c = bboxs_c[0].numpy()
        bbox_c = [int(x) for x in bbox_c]
        cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 4)
        cv2.rectangle(cv_image, (bbox[4], bbox[5]), (bbox[6], bbox[7]), (0, 255, 255), 4)
        cv2.circle(cv_image, (bbox_c[4], bbox_c[5]), 4, (0, 255, 255))
        cv2.imshow('test', cv_image)
        cv2.waitKey(2000)
        param = [params[0][0], params[1][0], params[2][0]]
        bbox = pts_trans_inv(bbox, param[0], param[1], param[2])
        bbox_c = pts_trans_inv(bbox_c, param[0], param[1], param[2])
        pil_image = Image.open(images_path[0])
        cv_image = np.array(pil_image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 4)
        cv2.rectangle(cv_image, (bbox[4], bbox[5]), (bbox[6], bbox[7]), (0, 255, 255), 4)
        cv2.circle(cv_image, (bbox_c[4], bbox_c[5]), 4, (0, 255, 255))
        cv2.imshow('test', cv_image)
        cv2.waitKey(2000)

# predict image folder with suffix '.jpg'
class DRDetection_predict_jpg_folder(Dataset):
    def __init__(self, root, scale_size=None):
        self.root = root
        self.scale_size = scale_size
        self.transform = transforms.Compose([
            PILColorJitter(),
            transforms.ToTensor(),
            # Lighting(alphastd=0.01, eigval=eigen_values, eigvec=eigen_values),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.img_list = glob(os.path.join(root, '*.jpg'))
        # img_list = [i.split('.')[0] for i in img_list]

    def __getitem__(self, item):
        image_path = self.img_list[item]
        print(image_path)
        img = Image.open(image_path)
        img, l, u, ratio = scale_image(img, self.scale_size)
        img = self.transform(img)
        return img, image_path, [l,u,ratio]

    def __len__(self):
        return len(self.img_list)

class DRDetection_predict_png_folder(Dataset):
    def __init__(self, root, scale_size=None):
        self.root = root
        self.scale_size = scale_size
        self.transform = transforms.Compose([
            PILColorJitter(),
            transforms.ToTensor(),
            # Lighting(alphastd=0.01, eigval=eigen_values, eigvec=eigen_values),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.img_list = glob(os.path.join(root, '*.png'))
        # img_list = [i.split('.')[0] for i in img_list]

    def __getitem__(self, item):
        image_path = self.img_list[item]
        print(image_path)
        img = Image.open(image_path)
        img, l, u, ratio = scale_image(img, self.scale_size)
        img = self.transform(img)
        return img, image_path, [l,u,ratio]

    def __len__(self):
        return len(self.img_list)

# predict single image with

def DRDetection_predict_single_image(imagepath, scale_size=None):
    transform = transforms.Compose([
        PILColorJitter(),
        transforms.ToTensor(),
        # Lighting(alphastd=0.01, eigval=eigen_values, eigvec=eigen_values),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(imagepath)
    img, l, u, ratio = scale_image(img, scale_size)
    img = transform(img)
    return img, imagepath, [l,u,ratio]



# check predicted fovea validation
def get_detect_fovea_array(predict_pts, gt_pts, gt_od_bboxs, thres=0.5):
    assert predict_pts.shape[0] == gt_pts.shape[0] == gt_od_bboxs.shape[0]
    length = predict_pts.shape[0]
    result = np.array([], dtype=int)
    bias = np.array([], dtype=float)
    for i in range(length):
        od_w = gt_od_bboxs[i][2] - gt_od_bboxs[i][0]
        od_h = gt_od_bboxs[i][3] - gt_od_bboxs[i][1]
        r = max(od_w, od_h)/2
        dis_thres = r * thres
        dis_fovea = math.sqrt(pow(gt_pts[i][0]-predict_pts[i][0], 2)+pow(gt_pts[i][1]-predict_pts[i][1],2))
        checked = 1 if (dis_fovea<dis_thres) else 0
        result = np.append(result, checked)
        bias = np.append(bias, dis_fovea/r)
    return result, bias

# check predict od validation
def get_detect_od_array(predict_od_bboxs, gt_od_bboxs, thres=0.7):
    assert predict_od_bboxs.shape[0] == gt_od_bboxs.shape[0]
    length = predict_od_bboxs.shape[0]
    result = np.array([], dtype=int)
    ious = np.array([], dtype=float)
    for i in range(length):
        iou = cal_iou(predict_od_bboxs[i], gt_od_bboxs[i])
        checked = 1 if (iou > thres) else 0
        result = np.append(result, checked)
        ious = np.append(ious, iou)
    return result, ious

def clamp(self, min_val, max_val, val):
    return max(min_val, min(max_val, val))

def get_random_bbox(bboxs, padding, size, type=1):
    if type == 1:
        min_x_b = bboxs[2]
        min_y_b = bboxs[3]
        max_x_b = bboxs[0]
        max_y_b = bboxs[1]
    elif type == 2:
        min_x_b = max(bboxs[2], bboxs[6])
        min_y_b = max(bboxs[3], bboxs[7])
        max_x_b = min(bboxs[0], bboxs[4])
        max_y_b = min(bboxs[1], bboxs[5])

    min_x_b =  max(0+padding, min_x_b-padding)
    min_y_b =  max(0+padding, min_y_b-padding)
    max_x_b =  min(size - padding, max_x_b+padding)
    max_y_b =  min(size - padding, max_y_b+padding)
    x_center = random.randint(min_x_b, max_x_b)
    y_center = random.randint(min_y_b, max_y_b)
    return x_center - padding, y_center - padding, x_center + padding, y_center + padding


'''
dataset related
'''

ds_type = ['od', 'fovea', 'od_fovea']

class DRDetectionDS_xml(Dataset):
    def __init__(self, root, ann_root, crop, size, type, scale_size=None):
        super(DRDetectionDS_xml, self).__init__()
        self.root = root
        self.ann_root = ann_root
        self.scale_size = scale_size
        self.crop = crop
        self.size = size
        self.ratio = size/crop
        self.padding = crop//2
        self.type = type
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
        if self.type == ds_type[0]:
            return anns, pts, pts_c[:4]
        elif self.type == ds_type[1]:
            return anns, pts, pts_c[4:6]
        else:
            return anns, pts, pts_c[:6]



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
        # assert len(bboxs) == 4
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

    def __get_random_bbox_constrained(self, bboxs, padding, size):
        return get_random_bbox(bboxs, padding, size, 1)


    def __getitem__(self, item):
        anns = self.ann_info_list[item]
        image_path = os.path.join(self.root, self.data_list[item]+'.png')
        img = Image.open(image_path)
        # l,u,r,b = self.__get_random_bbox(self.padding, self.size)
        l, u, r, b = self.__get_random_bbox_constrained(self.bboxs_list[item], self.padding, self.size)
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

class DRDetectionDS_od_xml(DRDetectionDS_xml):
    def __init__(self, root, ann_root, crop, size, scale_size=None):
        super(DRDetectionDS_od_xml, self).__init__(root, ann_root, crop, size, ds_type[0],scale_size)

class DRDetectionDS_fovea_xml(DRDetectionDS_xml):
    def __init__(self, root, ann_root, crop, size, scale_size=None):
        super(DRDetectionDS_fovea_xml, self).__init__(root, ann_root, crop, size, ds_type[1],scale_size)

class DRDetectionDS_od_and_fovea_xml(DRDetectionDS_xml):
    def __init__(self, root, ann_root, crop, size, scale_size=None):
        super(DRDetectionDS_od_and_fovea_xml, self).__init__(root, ann_root, crop, size, ds_type[2],scale_size)


class DRDetectionDS_predict_xml(Dataset):
    def __init__(self, root, ann_root, type, scale_size=None):
        super(DRDetectionDS_predict_xml, self).__init__()
        self.root = root
        self.ann_root = ann_root
        self.scale_size = scale_size
        self.type = type
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
        if self.type == ds_type[0]:
            return anns, pts, pts_c[:4]
        elif self.type == ds_type[1]:
            return anns, pts, pts_c[4:6]
        else:
            return anns, pts, pts_c[:6]

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

class DRDetectionDS_od_predict_xml(DRDetectionDS_predict_xml):
    def __init__(self, root, ann_root, scale_size=None):
        super(DRDetectionDS_od_predict_xml, self).__init__(root, ann_root, ds_type[0], scale_size)

class DRDetectionDS_fovea_predict_xml(DRDetectionDS_predict_xml):
    def __init__(self, root, ann_root, scale_size=None):
        super(DRDetectionDS_fovea_predict_xml, self).__init__(root, ann_root, ds_type[1], scale_size)

class DRDetectionDS_od_and_fovea_predict_xml(DRDetectionDS_predict_xml):
    def __init__(self, root, ann_root, scale_size=None):
        super(DRDetectionDS_od_and_fovea_predict_xml, self).__init__(root, ann_root, ds_type[2], scale_size)



if __name__ == '__main__':
    # test_DRDetectionDS_predict()
    test_DRDetection_predict_raw()
    # test_DRDetectionDS_od_xml()
