from torch.utils.data import Dataset, DataLoader
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from glob import glob

class DRDetectionDS_xml(Dataset):
    def __init__(self, root, ann_root, scale_size=None):
        super(DRDetectionDS_xml, self).__init__()
        self.root = root
        self.ann_root = ann_root
        self.scale_size = scale_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
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
        for ann in ann_list:
            if ann in img_list:
                self.data_list.append(ann)
        for index in self.data_list:
            anns = self.__read_xml(index)
            self.ann_info_list.append(anns)


    def __read_xml(self, xml_file):
        xml_file = os.path.join(self.ann_root, xml_file+'.xml')
        tree = ET.parse(xml_file)
        anns = []
        for obj in tree.getiterator('object'):
            if obj.find('name').text == 'optic_disk' or obj.find('name').text == 'macular':
                ann = {}
                # ann['cls_id'] = obj.find('name').text
                ann['ordered_id'] = 1 if obj.find('name').text == 'optic_disk' else 2
                ann['bbox'] = [0] * 4
                xmin = obj.find('bndbox').find('xmin')
                ymin = obj.find('bndbox').find('ymin')
                xmax = obj.find('bndbox').find('xmax')
                ymax = obj.find('bndbox').find('ymax')
                ann['bbox'][0] = int(xmin.text)
                ann['bbox'][1] = int(ymin.text)
                ann['bbox'][2] = int(xmax.text)
                ann['bbox'][3] = int(ymax.text)
                ann['scale_ratio'] = 1
                ann['area'] = (int(ymax.text)-int(ymin.text)) * (int(xmax.text)-int(xmin.text))
                anns.append(ann)
        return anns

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

        return img, anns, image_path

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

def dataset_test():
    dataloader = torch.utils.data.DataLoader(DRDetectionDS_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
                                                           '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
                                                           700), collate_fn=coco_collate,
                                             shuffle=True, pin_memory=True)

    for i, (img, anns,_) in enumerate(dataloader):
        # pil_img = transforms.ToPILImage()(img[0])
        # pil_img.show()
        print(anns)


if __name__ == '__main__':
    dataset_test()