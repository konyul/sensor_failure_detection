# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

from mmcv.ops import nms as nms_mmcv
from mmdet3d.core.evaluation import bbox_overlaps
@BBOX_CODERS.register_module()
class MultiTaskBBoxCoder_iou(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def iou_nms(self, bboxes, scores, locs, iou_threshold):
        """
        Use localization score to guide NMS. Bboxes are sorted by locs, the final score is
        taking the max of scores of all suppressed bboxes.
        """
        import pdb;pdb.set_trace()
        _, keep = nms_mmcv(bboxes, locs, iou_threshold)
        keep_bboxes = bboxes[keep]
        overlaps = bbox_overlaps(keep_bboxes, bboxes)
        # find the suppressed bboxes for each kept bbox
        suppressed = overlaps > iou_threshold
        # accumulate suppression count
        suppressed = suppressed.long().cummax(0)[0].cumsum(0)
        # the real suppressed bboxes should be the ones that are suppressed exactly once
        suppressed = (suppressed == 1)
        span_scores = scores.view(1, overlaps.size(1)).repeat(overlaps.size(0), 1)
        span_scores[~suppressed] = scores.min() - 1
        # taking the max of the suppressed group
        keep_scores = span_scores.max(1)[0]
        # sort by scores, following tradition
        keep_scores, srt_idx = keep_scores.sort(descending=True)
        return keep[srt_idx], keep_scores


    def batched_iou_nms(self, bboxes, scores, locs, labels, iou_threshold, guide='rank'):
        """IoU guided NMS.
        
        Args:
            bboxes (Tensor): shape (n, 4)
            scores (Tensor): shape (n,), classification score
            locs (Tensor): shape(n,) localization score/iou score
            labels (Tensor): label of bboxes, help to do batched nms
            iou_threshold (float): iou threshold
            score_thr (float): filter scores belowing this number
            guide (str): decide how to use iou score to guide nms, 
                supported key words: none, rank, weight"""
        
        nms_bboxes = bboxes + (labels * (bboxes.max() + 1)).view(-1, 1)
        # 'rank' is the official iou guided nms
        if guide == 'rank':
            keep, keep_scores = self.iou_nms(nms_bboxes, scores, locs, iou_threshold)
            return bboxes[keep], keep_scores, locs[keep], labels[keep]
        # use the product of cls_score * iou_score as the nms_locs
        elif guide == 'weight':
            nms_locs = scores * locs
            keep, keep_scores = self.iou_nms(nms_bboxes, scores, nms_locs, iou_threshold)
            return bboxes[keep], keep_scores, locs[keep], labels[keep]
        # do not utilize iou_score in nms
        elif guide == 'none':
            _, keep = nms_mmcv(nms_bboxes, scores, iou_threshold)
            return bboxes[keep], scores[keep], locs[keep], labels[keep]
        else:
            raise RuntimeError('guide type not supported: {}'.format(guide))

    def decode_single(self, cls_scores, bbox_preds, iou_pred, task_ids):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        num_query = cls_scores.shape[0]

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        task_index = torch.gather(task_ids, 1, labels.unsqueeze(1)).squeeze()

        bbox_preds = bbox_preds[task_index * num_query + bbox_index]
        boxes3d = denormalize_bbox(bbox_preds, self.pc_range)

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            mask = (boxes3d[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (boxes3d[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = boxes3d[mask]
            scores = scores[mask]
            labels = labels[mask]
            iou_pred = iou_pred[mask]
            self.batched_iou_nms(boxes3d, scores, iou_pred, labels, iou_threshold, guide='rank')

        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        }
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        task_num = len(preds_dicts)

        pred_bbox_list, pred_logits_list, pred_iou_list, task_ids_list, rv_box_mask_lists = [], [], [], [] , []
        for task_id in range(task_num):
            task_pred_dict = preds_dicts[task_id][0]
            task_pred_bbox = [task_pred_dict['center'][-1], task_pred_dict['height'][-1],
                              task_pred_dict['dim'][-1], task_pred_dict['rot'][-1]]
            if 'vel' in task_pred_dict:
                task_pred_bbox.append(task_pred_dict['vel'][-1])
            if 'iou' in task_pred_dict:
                task_pred_iou = task_pred_dict['iou'][-1]
                
            task_pred_bbox = torch.cat(task_pred_bbox, dim=-1)
            task_pred_logits = task_pred_dict['cls_logits'][-1]
            pred_bbox_list.append(task_pred_bbox)
            pred_logits_list.append(task_pred_logits)
            pred_iou_list.append(task_pred_iou)

            if "rv_box_mask" in task_pred_dict:
                rv_box_mask_lists.append(task_pred_dict["rv_box_mask"])
            else:
                rv_box_mask_lists.append(task_pred_dict["cls_logits"].new_ones(task_pred_dict["cls_logits"].shape[1], 6,
                                                                               task_pred_dict["cls_logits"].shape[
                                                                                   2]).to(torch.bool))

            task_ids = task_pred_logits.new_ones(task_pred_logits.shape).int() * task_id
            task_ids_list.append(task_ids)

        all_pred_logits = torch.cat(pred_logits_list, dim=-1)  # bs * nq * 10
        all_pred_bbox = torch.cat(pred_bbox_list, dim=1)  # bs * (task nq) * 10
        all_pred_iou = torch.cat(pred_iou_list,dim =-1)
        all_task_ids = torch.cat(task_ids_list, dim=-1)  # bs * nq * 10
        all_rv_box_masks = torch.cat(rv_box_mask_lists, dim=-1)

        batch_size = all_pred_logits.shape[0]
        predictions_list = []
        for i in range(batch_size):
            rv_box_mask = all_rv_box_masks[i].sum(dim=0) != 0
            if rv_box_mask.shape[0] != all_pred_bbox[i].shape[0]:
                box_mask = torch.cat([torch.ones_like(rv_box_mask), rv_box_mask])
            else:
                box_mask = rv_box_mask

            pred_logits = all_pred_logits[i][box_mask]
            pred_bbox = all_pred_bbox[i][box_mask]
            pred_iou = all_pred_iou[i][box_mask]
            task_ids = all_task_ids[i][box_mask]

            predictions_list.append(
                self.decode_single(pred_logits, pred_bbox, pred_iou, task_ids))
        return predictions_list



@BBOX_CODERS.register_module()
class MultiTaskBBoxCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, task_ids):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        num_query = cls_scores.shape[0]

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        task_index = torch.gather(task_ids, 1, labels.unsqueeze(1)).squeeze()

        bbox_preds = bbox_preds[task_index * num_query + bbox_index]
        boxes3d = denormalize_bbox(bbox_preds, self.pc_range)

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            mask = (boxes3d[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (boxes3d[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = boxes3d[mask]
            scores = scores[mask]
            labels = labels[mask]

        predictions_dict = {
            'bboxes': boxes3d,
            'scores': scores,
            'labels': labels
        }
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """

        task_num = len(preds_dicts)

        pred_bbox_list, pred_logits_list, task_ids_list, rv_box_mask_lists = [], [], [], []
        for task_id in range(task_num):
            task_pred_dict = preds_dicts[task_id][0]
            task_pred_bbox = [task_pred_dict['center'][-1], task_pred_dict['height'][-1],
                              task_pred_dict['dim'][-1], task_pred_dict['rot'][-1]]
            if 'vel' in task_pred_dict:
                task_pred_bbox.append(task_pred_dict['vel'][-1])
            task_pred_bbox = torch.cat(task_pred_bbox, dim=-1)
            task_pred_logits = task_pred_dict['cls_logits'][-1]
            pred_bbox_list.append(task_pred_bbox)
            pred_logits_list.append(task_pred_logits)

            if "rv_box_mask" in task_pred_dict:
                rv_box_mask_lists.append(task_pred_dict["rv_box_mask"])
            else:
                rv_box_mask_lists.append(task_pred_dict["cls_logits"].new_ones(task_pred_dict["cls_logits"].shape[1], 6,
                                                                               task_pred_dict["cls_logits"].shape[
                                                                                   2]).to(torch.bool))

            task_ids = task_pred_logits.new_ones(task_pred_logits.shape).int() * task_id
            task_ids_list.append(task_ids)

        all_pred_logits = torch.cat(pred_logits_list, dim=-1)  # bs * nq * 10
        all_pred_bbox = torch.cat(pred_bbox_list, dim=1)  # bs * (task nq) * 10
        all_task_ids = torch.cat(task_ids_list, dim=-1)  # bs * nq * 10
        all_rv_box_masks = torch.cat(rv_box_mask_lists, dim=-1)

        batch_size = all_pred_logits.shape[0]
        predictions_list = []
        for i in range(batch_size):
            rv_box_mask = all_rv_box_masks[i].sum(dim=0) != 0
            if rv_box_mask.shape[0] != all_pred_bbox[i].shape[0]:
                box_mask = torch.cat([torch.ones_like(rv_box_mask), rv_box_mask])
            else:
                box_mask = rv_box_mask

            pred_logits = all_pred_logits[i][box_mask]
            pred_bbox = all_pred_bbox[i][box_mask]
            task_ids = all_task_ids[i][box_mask]

            predictions_list.append(
                self.decode_single(pred_logits, pred_bbox, task_ids))
        return predictions_list
