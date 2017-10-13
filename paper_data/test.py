import sys
sys.path.append('../')

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage

from data_xml import DRDetectionDS_xml, coco_collate

dataloader = torch.utils.data.DataLoader(DRDetectionDS_xml('/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
                                                           '/home/weidong/code/github/mask_rcnn_pytorch/paper_data/data/data',
                                                           512), collate_fn=coco_collate,
                                             shuffle=True, pin_memory=True)

trans = ToPILImage()

# for i, (img, anns, image_path) in enumerate(dataloader):
#     # pil_img = transforms.ToPILImage()(img[0])
#     # pil_img.show()
#     bb1 = anns[0][0]['bbox']
#     bb2 = anns[0][1]['bbox']
#     im2show = np.copy(np.array(trans(img[0])))
#     cv2.rectangle(im2show, (int(bb1[0]), int(bb1[1])), (int(bb1[2]), int(bb1[3])), (200, 200, 0), 4)
#     cv2.rectangle(im2show, (int(bb2[0]), int(bb2[1])), (int(bb2[2]), int(bb2[3])), (200, 200, 0), 4)
#     cv2.imshow('test', im2show)
#     cv2.waitKey(2000)
#     print(anns)


for i, (img, anns, image_path) in enumerate(dataloader):
    # pil_img = transforms.ToPILImage()(img[0])
    # pil_img.show()
    raw = cv2.imread(image_path[0])
    bb1 = anns[0][0]['bbox']
    bb2 = anns[0][1]['bbox']
    im2show = np.copy(raw)
    cv2.rectangle(im2show, (int(bb1[0]), int(bb1[1])), (int(bb1[2]), int(bb1[3])), (200, 200, 0), 4)
    cv2.rectangle(im2show, (int(bb2[0]), int(bb2[1])), (int(bb2[2]), int(bb2[3])), (200, 200, 0), 4)
    cv2.imshow('test', im2show)
    cv2.waitKey(2000)
    print(anns)
