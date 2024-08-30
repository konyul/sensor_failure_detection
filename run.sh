#tools/dist_train.sh projects/configs/moad_voxel0100_r50_800x320_cbgs.py  4
# work_dir=work_dirs/moad_voxel0100_r50_800x320_cbgs/1_5
# bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_concat_img_masking_small.py 4 --work-dir $work_dir 
# for failures in 'occlusion'
# do
#   bash tools/dist_test.sh projects/configs/failure/moad_voxel0100_r50_800x320_cbgs_$failures.py $work_dir/epoch_20.pth 4 --eval bbox
# done

# bash tools/dist_test.sh projects/configs/moad_voxel0100_r50_800x320_cbgs.py $work_dir/epoch_20.pth 4 --eval bbox

# work_dir=work_dirs/moad_ky_voxel0100_r50_800x320_cbgs/lid_mask_rat05_freq_025/20240819-135245
# # bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_concat_img_masking_small.py 4 --work-dir $work_dir 
# for failures in 'lidar_drop' 'camera_drop' 'limited_fov' 'object_failure' 'beam_reduction' 'occlusion'
# do
#   bash tools/dist_test.sh projects/configs/failure/moad_voxel0100_r50_800x320_cbgs_$failures.py $work_dir/epoch_20.pth 4 --eval bbox
# done

# bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_concat_img_masking_small.py 4 --work-dir $work_dir 
for failures in 'lidar_drop' 'camera_drop' 'limited_fov' 'object_failure' 'beam_reduction' 'occlusion'
do
  bash tools/dist_test.sh projects/configs/failure_0075/moad_voxel0075_vov_1600x640_cbgs_$failures.py ckpts/meformer_voxel0075_vov_1600x640_cbgs.pth 4 --eval bbox
done
