import xml.etree.ElementTree as ET
import os
from predict_common import scale_image, pts_trans
from PIL import Image
import numpy as np

from glob import glob

data_root = '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/ex'
p_data_root = '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/ex_512'
os.makedirs(p_data_root, exist_ok=True)


def single_processing(image_file, data_root, p_data_root):
    image_file = image_file
    xml_file = os.path.join(data_root, os.path.basename(image_file).split('.')[0] + '.xml')
    if not os.path.exists(xml_file):
        return
    p_data_root = p_data_root
    p_image_file = os.path.join(p_data_root, os.path.basename(image_file).split('.')[0] + '.png')
    p_xml_file = os.path.join(p_data_root, os.path.basename(image_file).split('.')[0] + '.xml')
    tree = ET.parse(xml_file)
    pil_img = Image.open(image_file)
    pil_img, l, u, ratio = scale_image(pil_img, 512)
    for obj in tree.getiterator('object'):
        if (obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc' or obj.find(
                'name').text == 'optic-disc' or obj.find('name').text == 'macular'):
            ann = {}
            # ann['cls_id'] = obj.find('name').text
            ann['ordered_id'] = 1 if (
            obj.find('name').text == 'optic_disk' or obj.find('name').text == 'optic_disc') else 2
            # ann['bbox'] = [0] * 4
            xmin = obj.find('bndbox').find('xmin')
            ymin = obj.find('bndbox').find('ymin')
            xmax = obj.find('bndbox').find('xmax')
            ymax = obj.find('bndbox').find('ymax')

            tmp = np.array([], dtype=int)
            tmp = np.append(tmp, np.array([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)]))
            tmp = pts_trans(tmp, l, u, ratio)
            tmp = [int(i) for i in tmp]

            obj.find('bndbox').find('xmin').text = str(tmp[0])
            obj.find('bndbox').find('ymin').text = str(tmp[1])
            obj.find('bndbox').find('xmax').text = str(tmp[2])
            obj.find('bndbox').find('ymax').text = str(tmp[3])
    pil_img.save(p_image_file)
    tree.write(p_xml_file)


images_list = glob(os.path.join(data_root, '*.jpg'))
for image_file in images_list:
    print(image_file)
    single_processing(image_file, data_root, p_data_root)




