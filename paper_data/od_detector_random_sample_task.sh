#CUDA_VISIBLE_DEVICES=0,1,2,3 python od_detector.py --phase train
CUDA_VISIBLE_DEVICES=0,1,2,3 python od_detector_random_sample.py --phase test --weight /home/weidong/code/github/mask_rcnn_pytorch/paper_data/output/emei_od_and_fovea_detection_train_20171102101830_rsn34_od_and_fovea_detection/emei_od_and_fovea_detection_rsn34_00334_best.pth
