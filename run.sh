



# tools/dist_test.sh projects/configs/moad_voxel0100_r50_800x320_cbgs.py ./work_dirs/moad_voxel0100_r50_800x320_cbgs/1_5/epoch_20.pth 4 --eval bbox

# tools/dist_test.sh projects/configs/moad_voxel0100_r50_800x320_cbgs.py ./work_dirs/moad_voxel0100_r50_800x320_cbgs/1_5/epoch_16.pth 4 --eval bbox

# tools/dist_test.sh projects/configs/moad_voxel0100_r50_800x320_cbgs.py ./work_dirs/moad_voxel0100_r50_800x320_cbgs/1_5/epoch_12.pth 4 --eval bbox

# tools/dist_test.sh projects/configs/moad_voxel0100_r50_800x320_cbgs.py ./work_dirs/moad_voxel0100_r50_800x320_cbgs/1_5/epoch_8.pth 4 --eval bbox

tools/dist_test.sh projects/configs/e2e_moad_r50.py ./work_dirs/iou_2stage/res_0100/0909_e2e_b4_lr_0.0001_delete_dn_smoothL1_sigmoid_pred/20240909-124809/epoch_20.pth 4 --eval bbox

## test
## multi_task_bbox_coder.py


## train
## meformer_head.py
## centernet_loss.py
