import torch

inval_zero = 1e-100

def cal_iou(input, target):
    inval = False
    if (input[0] >= input[2]) or (input[1] >= input[3]):
        return inval_zero
    if (target[0] >= target[2]) or (target[1] >= target[3]):
        return inval_zero

    return 0

def test_cal_iou():
    input = torch.FloatTensor([10,10,200,299])
    target = torch.FloatTensor([23, 32, 435,344])
    cal_iou(input, target)

test_cal_iou()