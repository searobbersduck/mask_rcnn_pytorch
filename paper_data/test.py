from PIL import Image
import numpy as np
import random
import os
import xml.etree.ElementTree as ET

from predict_common import get_random_bbox

image_file = '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/dr_ann/690_dr_2_dme_0_512.png'
ann_path = '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/dr_ann'

mask_file = os.path.join(ann_path, os.path.basename(image_file).split('.')[0]+'.xml')

crop = 448
size = 512
padding = 224

def get_bbox(padding, size):
    x_center = random.randint(0+padding, size-padding)
    y_center = random.randint(0+padding, size-padding)
    return x_center-padding, y_center-padding, x_center+padding, y_center+padding

image = Image.open(image_file)

def read_xml(xml_file):
    # xml_file = os.path.join(self.ann_root, xml_file + '.xml')
    tree = ET.parse(xml_file)
    anns = []
    pts = np.array([], dtype=int)
    pts_c = np.array([], dtype=int)
    for obj in tree.getiterator('object'):
        if (obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc' or obj.find(
                'name').text == 'optic-disc'):
            ann = {}
            # ann['cls_id'] = obj.find('name').text
            ann['ordered_id'] = 1 if (
            obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc') else 2
            # ann['bbox'] = [0] * 4
            xmin = obj.find('bndbox').find('xmin')
            ymin = obj.find('bndbox').find('ymin')
            xmax = obj.find('bndbox').find('xmax')
            ymax = obj.find('bndbox').find('ymax')
            ann['bbox'] = np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
            ann['scale_ratio'] = 1
            ann['area'] = (int(ymax.text) - int(ymin.text)) * (int(xmax.text) - int(xmin.text))
            anns.append(ann)
            pts = np.append(pts, np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)]))
            pts_c = np.append(pts_c, np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)]))
    for obj in tree.getiterator('object'):
        if obj.find('name').text == 'macular':
            ann = {}
            # ann['cls_id'] = obj.find('name').text
            ann['ordered_id'] = 1 if (
            obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc' or obj.find(
                'name').text == 'optic-disc') else 2
            # ann['bbox'] = [0] * 4
            xmin = obj.find('bndbox').find('xmin')
            ymin = obj.find('bndbox').find('ymin')
            xmax = obj.find('bndbox').find('xmax')
            ymax = obj.find('bndbox').find('ymax')
            ann['bbox'] = np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
            ann['scale_ratio'] = 1
            ann['area'] = (int(ymax.text) - int(ymin.text)) * (int(xmax.text) - int(xmin.text))
            anns.append(ann)
            pts = np.append(pts, np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)]))
            pts_c = np.append(pts_c, np.array(
                [(int(xmin.text) + int(xmax.text)) // 2, (int(ymin.text) + int(ymax.text)) // 2]))
    return anns, pts, pts_c[:6]

_,bboxs,_ = read_xml(mask_file)

def bbox_trans(bboxs, l, u, ratio):
    assert len(bboxs) == 8
    pts = np.array([], dtype=int)
    for i in range(len(bboxs)):
        if i%2 == 0:
            pts = np.append(pts, int((bboxs[i]-l)*ratio))
        else:
            pts = np.append(pts, int((bboxs[i]-u)*ratio))
    return pts

for i in range(2):
    # l,u,r,b = get_bbox(padding, size)
    l,u,r,b = get_random_bbox(bboxs, padding, size, 1)
    cropped_image = image.crop((l,u,r,b))
    cropped_bboxs = bbox_trans(bboxs, l, u, size/crop)
    print('crop and resized image bounding box: {}'.format(cropped_bboxs))
    cropped_image.show()




