import torch


def cal_iou(input, target):
    return 0

def test_cal_iou():
    input = torch.FloatTensor([10,10,200,299])
    target = torch.FloatTensor([23, 32, 435,344])
    cal_iou(input, target)

test_cal_iou()