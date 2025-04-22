import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

class PanopticEvaluatorKITTI:
    def __init__(self, iou_thresh=0.5, thing_ids=None, ignore_label=0):
        """
        针对 KITTI 数据集的全景评价器

        参数:
            iou_thresh: IoU 阈值，通常设为 0.5
            thing_ids: 属于实例（thing）类别的语义标签集合，例如 {10,11,12,13,14,15,16,17}
            ignore_label: 被忽略的标签（例如 0）
        """
        if thing_ids is None:
            thing_ids = {10, 11, 12, 13, 14, 15, 16, 17}
        self.iou_thresh = iou_thresh
        self.thing_ids = thing_ids
        self.ignore_label = ignore_label
        # 用于区分类别内部不同实例的编码常数，一般取一个足够大的值
        self.max_ins = 50  # 约 1e6
        self.reset()

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.iou_sum = 0.0

    def _combine_panoptic(self, semseg, instance):
        """
        根据语义分割和实例分割生成全景 GT 图：
          - 对于属于 thing 类别（在 self.thing_ids 中）的像素，
            编码方式为： semseg * max_ins + instance
          - 对于 stuff 类别，直接使用语义标签；
          - 对于 ignore 标签（如 0）统一设置为 -1，不参与评价。
        参数:
            semseg: (H, W) 的 numpy 数组，GT 语义分割
            instance: (H, W) 的 numpy 数组，GT 实例分割
        返回:
            panoptic: (H, W) 的 numpy 数组，合成后的全景 GT 编码
        """
        semseg = semseg.astype(np.int32)
        instance = instance.astype(np.int32)
        # 对于 thing 类别按照编码规则合并；否则直接用 semseg 值
        panoptic = np.where(np.isin(semseg, list(self.thing_ids)),
                            semseg * self.max_ins + instance,
                            semseg)
        # ignore 部分统一设置为 -1
        panoptic[semseg == self.ignore_label] = -1
        return panoptic

    def _compute_iou(self, mask1, mask2):
        """
        计算两个二值 mask 的交并比（IoU）
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def evaluate_image(self, pred_panoptic, gt_semseg, gt_instance):
        """
        对单张图像进行评价：
          - 先将 GT 的 semseg 与 instance 合成全景 GT 图
          - 遍历 GT 中的每个区域（除去 ignore 区域），在预测结果中寻找相同类别（根据编码）IoU
            最大且大于阈值的区域作为匹配（TP）；未匹配计 FN，而预测中多出的区域计 FP。
        返回:
          tp, fp, fn, iou_sum
        """
        gt_panoptic = self._combine_panoptic(gt_semseg, gt_instance)
        # 提取所有区域，不包括 ignore（即 -1）
        gt_ids = np.unique(gt_panoptic)
        pred_ids = np.unique(pred_panoptic)
        gt_ids = gt_ids[gt_ids != -1]
        pred_ids = pred_ids[pred_ids != -1]

        # 构造区域对应的二值 mask
        gt_masks = {gid: (gt_panoptic == gid) for gid in gt_ids}
        pred_masks = {pid: (pred_panoptic == pid) for pid in pred_ids}

        matched_preds = set()
        tp = 0
        iou_sum = 0.0
        # 对于每个 GT 区域，寻找最佳匹配的预测区域（类别必须一致）
        for gt_id, gt_mask in gt_masks.items():
            # 如果 gt_id 是 thing 类别，则类别为 gt_id // max_ins，否则为 gt_id 自身
            gt_cat = (gt_id // self.max_ins) if gt_id >= self.max_ins else gt_id
            best_iou = 0.0
            best_pred = None
            for pred_id, pred_mask in pred_masks.items():
                pred_cat = (pred_id // self.max_ins) if pred_id >= self.max_ins else pred_id
                if pred_cat != gt_cat:
                    continue
                iou = self._compute_iou(gt_mask, pred_mask)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_id
            if best_iou >= self.iou_thresh and best_pred is not None:
                tp += 1
                iou_sum += best_iou
                matched_preds.add(best_pred)
        fp = len(pred_masks) - len(matched_preds)
        fn = len(gt_masks) - tp
        return tp, fp, fn, iou_sum

    def add_image(self, pred_panoptic, gt_semseg, gt_instance):
        """
        对单张图像，累积评价指标
        """
        tp, fp, fn, iou_sum = self.evaluate_image(pred_panoptic, gt_semseg, gt_instance)
        self.TP += tp
        self.FP += fp
        self.FN += fn
        self.iou_sum += iou_sum

    def evaluate(self):
        """
        根据累计的 TP、FP、FN、iou_sum 计算评价指标：
            SQ = iou_sum / TP
            RQ = TP / (TP + 0.5 * (FP + FN))
            PQ = SQ * RQ
        """
        epsilon = 1e-10
        if self.TP == 0:
            sq = 0.0
            rq = 0.0
        else:
            sq = self.iou_sum / (self.TP + epsilon)
            rq = self.TP / (self.TP + 0.5 * (self.FP + self.FN) + epsilon)
        pq = sq * rq
        return {"pq": pq, "sq": sq, "rq": rq, "iou_sum": self.iou_sum, "tp": self.TP, "fp": self.FP, "fn": self.FN}
