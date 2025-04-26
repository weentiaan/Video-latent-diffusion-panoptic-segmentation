import numpy as np

class KITTIPanopticEvaluator:
    """
    KITTI panoptic evaluator  (prediction = semantic + instance)

    Args
    ----
    thing_ids     : set(int)    需要区分实例的类别 ID
    ignore_label  : int         忽略像素的语义标签
    iou_thresh    : float       IoU 阈值（Panoptic 定义里固定 0.5）
    max_ins       : int         用于编码 panoptic ID 的乘数，需足够大
    """

    def __init__(self,
                 thing_ids   = {10,11,12,13,14,15,16,17},
                 ignore_label=0,
                 iou_thresh  =0.5,
                 max_ins     =1 << 20):
        self.thing_ids    = set(thing_ids)
        self.ignore_label = ignore_label
        self.iou_thresh   = iou_thresh
        self.max_ins      = max_ins      # 约 1e6，可保证不同实例不冲突
        self.reset()

    # ------------------------------------------------------------------ utils
    def _to_panoptic(self, sem, ins):
        """
        将语义 + 实例映射成 panoptic ID:
            stuff →  语义标签
            thing →  sem * max_ins + ins
        忽略像素用 -1 表示，不参与评价
        """
        sem  = sem.astype(np.int64)
        ins  = ins.astype(np.int64)
        pan  = np.where(np.isin(sem, list(self.thing_ids)),
                        sem * self.max_ins + ins,
                        sem)
        pan[sem == self.ignore_label] = -1
        return pan

    @staticmethod
    def _iou(mask1, mask2):
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return 0.0 if union == 0 else inter / union

    # ----------------------------------------------------------- public API
    def reset(self):
        self.TP = self.FP = self.FN = 0
        self.iou_sum = 0.0

    def add_image(self, pred_seg, pred_ins, gt_semseg, gt_instance):
        """
        Parameters
        ----------
        pred_seg      : (H,W) 预测语义标签
        pred_ins      : (H,W) 预测实例 ID (0 = void / background)
        gt_semseg     : (H,W) GT 语义标签
        gt_instance   : (H,W) GT 实例 ID
        """
        pred_pan = self._to_panoptic(pred_seg, pred_ins)
        gt_pan   = self._to_panoptic(gt_semseg, gt_instance)

        # 枚举 GT 区域并匹配预测
        gt_ids, pred_ids = np.unique(gt_pan), np.unique(pred_pan)
        gt_ids   = gt_ids[  gt_ids != -1]
        pred_ids = pred_ids[pred_ids != -1]

        gt_masks   = {gid: gt_pan   == gid for gid in gt_ids}
        pred_masks = {pid: pred_pan == pid for pid in pred_ids}

        matched_pred = set()
        for gid, gmask in gt_masks.items():
            gcat = (gid // self.max_ins) if gid >= self.max_ins else gid
            best_iou, best_pid = 0.0, None
            for pid, pmask in pred_masks.items():
                pcat = (pid // self.max_ins) if pid >= self.max_ins else pid
                if pcat != gcat:                       # 类别需一致
                    continue
                iou = self._iou(gmask, pmask)
                if iou > best_iou:
                    best_iou, best_pid = iou, pid
            if best_iou >= self.iou_thresh:
                self.TP       += 1
                self.iou_sum  += best_iou
                matched_pred.add(best_pid)

        self.FP += len(pred_masks) - len(matched_pred)
        self.FN += len(gt_masks)   - self.TP   + (len(matched_pred) - len(pred_masks))

    def evaluate(self):
        """
        Returns
        -------
        dict{ 'pq', 'sq', 'rq', 'tp', 'fp', 'fn', 'iou_sum'}
        """
        if self.TP == 0:
            sq = rq = pq = 0.0
        else:
            sq = self.iou_sum / self.TP
            rq = self.TP / (self.TP + 0.5 * (self.FP + self.FN))
            pq = sq * rq
        return dict(pq=pq, sq=sq, rq=rq,
                    tp=self.TP, fp=self.FP, fn=self.FN,
                    iou_sum=self.iou_sum)
