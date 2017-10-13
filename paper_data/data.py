from torch.utils.data import Dataset, DataLoader
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

class DRDetectionDS(torch.utils.data.Dataset):
    def __init__(self, root, config, scale_size=None):
        super(DRDetectionDS, self).__init__()
        self.root = root
        self.config = config
        self.scale_size = scale_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.data_list = []
        with open(config, 'r') as f:
            self.data_list = f.readlines()

    def __getitem__(self, item):
        data = self.data_list[item]
        anns = []
        vec = data.split('|')
        image_path = os.path.join(self.root, vec[0])
        for i in range(1, len(vec)):
            str = vec[i]
            info_vec = str.split(' ')
            ann = {}
            ann['ordered_id'] = int(info_vec[0])
            ann['bbox'] = [0] * 4
            ann['bbox'][0] = int(info_vec[1])
            ann['bbox'][1] = int(info_vec[2])
            ann['bbox'][2] = int(info_vec[1]) + int(info_vec[3])
            ann['bbox'][3] = int(info_vec[2]) + int(info_vec[4])
            ann['scale_ratio'] = 1
            ann['area'] = int(info_vec[3])*int(info_vec[4])
            anns.append(ann)

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
    dataloader = torch.utils.data.DataLoader(DRDetectionDS('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
                                                           '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/detection.txt',
                                                           700), collate_fn=coco_collate,
                                             shuffle=True, pin_memory=True)

    for i, (img, anns, _) in enumerate(dataloader):
        # pil_img = transforms.ToPILImage()(img[0])
        # pil_img.show()
        print(anns)


if __name__ == '__main__':
    dataset_test()