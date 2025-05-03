"""
Author: (Adapted from KITTI evaluator)
Cityscapes panoptic evaluator
"""

import numpy as np

class CityscapesPanopticEvaluator:
    """
    Cityscapes panoptic evaluator (prediction = semantic + instance)

    Args
    ----
    thing_ids     : set(int)    需要区分实例的类别 ID (Cityscapes 'thing' 类别)
    ignore_label  : int         忽略像素的语义标签
    iou_thresh    : float       IoU 阈值（Panoptic 定义里固定 0.5）
    max_ins       : int         用于编码 panoptic ID 的乘数，需足够大
    """

    def __init__(self,
                 thing_ids   = {11, 12, 13, 14, 15, 16, 17, 18},  # Cityscapes 'thing' 类别 ID
                 ignore_label = 0,
                 iou_thresh  = 0.5,
                 max_ins     = 1 << 20):
        self.thing_ids    = set(thing_ids)
        self.ignore_label = ignore_label
        self.iou_thresh   = iou_thresh
        self.max_ins      = max_ins      # 约 1e6，可保证不同实例不冲突
        self.reset()

    # ------------------------------------------------------------------ utils
    def _to_panoptic(self, sem, ins):
        """
        将语义 + 实例映射成 panoptic ID:
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

    # ----------------------------------------------------------- public API
    def reset(self):
        self.TP = self.FP = self.FN = 0
        self.iou_sum = 0.0
        # 添加类别级别的评估
        self.TP_per_class = {}
        self.FP_per_class = {}
        self.FN_per_class = {}
        self.iou_sum_per_class = {}

    def add_image(self, pred_seg, gt_semseg, gt_instance=None):
        """
        Parameters
        ----------
        pred_seg      : (H,W) 预测的全景分割结果
        gt_semseg     : (H,W) GT 语义标签
        gt_instance   : (H,W) 或 None, GT 实例 ID (可选参数)
        """
        # 如果没有提供gt_instance，创建一个简单的实例映射
        if gt_instance is None:
            gt_instance = np.zeros_like(gt_semseg)
            # 为thing类别创建唯一实例ID
            for sem_id in self.thing_ids:
                if sem_id in np.unique(gt_semseg):
                    # 为每个连通区域分配唯一ID
                    mask = gt_semseg == sem_id
                    from scipy import ndimage
                    labeled, num_features = ndimage.label(mask)
                    # 只在mask区域内使用labeled值
                    gt_instance[mask] = labeled[mask]
                    
        # 构建GT的全景分割
        gt_pan = self._to_panoptic(gt_semseg, gt_instance)
        
        # 从预测的全景分割中提取语义和实例信息
        # 这里假设pred_seg已经包含全景分割结果，不需要再组合
        pred_pan = pred_seg.copy()
        pred_pan[gt_semseg == self.ignore_label] = -1  # 忽略区域不参与评估

        # 枚举 GT 区域并匹配预测
        gt_ids, pred_ids = np.unique(gt_pan), np.unique(pred_pan)
        gt_ids   = gt_ids[  gt_ids != -1]
        pred_ids = pred_ids[pred_ids != -1]

        gt_masks   = {gid: gt_pan   == gid for gid in gt_ids}
        pred_masks = {pid: pred_pan == pid for pid in pred_ids}

        matched_pred = set()
        for gid, gmask in gt_masks.items():
            # 获取语义类别
            gcat = (gid // self.max_ins) if gid >= self.max_ins else gid
            
            # 初始化类别级别统计
            if gcat not in self.TP_per_class:
                self.TP_per_class[gcat] = 0
                self.FP_per_class[gcat] = 0
                self.FN_per_class[gcat] = 0
                self.iou_sum_per_class[gcat] = 0.0
                
            best_iou, best_pid = 0.0, None
            for pid, pmask in pred_masks.items():
                pcat = (pid // self.max_ins) if pid >= self.max_ins else pid
                if pcat != gcat:                       # 类别需一致
                    continue
                iou = self._iou(gmask, pmask)
                if iou > best_iou:
                    best_iou, best_pid = iou, pid
            
            if best_iou >= self.iou_thresh:
                self.TP += 1
                self.iou_sum += best_iou
                matched_pred.add(best_pid)
                
                # 类别级别统计
                self.TP_per_class[gcat] += 1
                self.iou_sum_per_class[gcat] += best_iou

        self.FP += len(pred_masks) - len(matched_pred)
        self.FN += len(gt_masks) - self.TP
        
        # 统计类别级别的 FP 和 FN
        for pid in pred_ids:
            pcat = (pid // self.max_ins) if pid >= self.max_ins else pid
            if pid not in matched_pred:
                if pcat not in self.FP_per_class:
                    self.FP_per_class[pcat] = 0
                self.FP_per_class[pcat] += 1
                
        for gid in gt_ids:
            gcat = (gid // self.max_ins) if gid >= self.max_ins else gid
            if gcat not in self.FN_per_class:
                self.FN_per_class[gcat] = 0
            self.FN_per_class[gcat] += 1
        
        # 更正 FN 计数
        for gcat in self.TP_per_class.keys():
            if gcat in self.FN_per_class:
                self.FN_per_class[gcat] -= self.TP_per_class[gcat]

    def evaluate(self):
        """
        Returns
        -------
        dict{ 'pq', 'sq', 'rq', 'tp', 'fp', 'fn', 'iou_sum', 'per_class'}
        """
        if self.TP == 0:
            sq = rq = pq = 0.0
        else:
            sq = self.iou_sum / self.TP
            rq = self.TP / (self.TP + 0.5 * (self.FP + self.FN))
            pq = sq * rq
            
        # 计算类别级别的指标
        per_class = {}
        for cat in self.TP_per_class.keys():
            tp = self.TP_per_class.get(cat, 0)
            fp = self.FP_per_class.get(cat, 0)
            fn = self.FN_per_class.get(cat, 0)
            iou_sum = self.iou_sum_per_class.get(cat, 0.0)
            
            if tp == 0:
                cat_sq = cat_rq = cat_pq = 0.0
            else:
                cat_sq = iou_sum / tp
                cat_rq = tp / (tp + 0.5 * (fp + fn))
                cat_pq = cat_sq * cat_rq
                
            per_class[int(cat)] = {
                'pq': cat_pq, 
                'sq': cat_sq, 
                'rq': cat_rq,
                'tp': tp, 
                'fp': fp, 
                'fn': fn
            }
            
        # 分离 'thing' 和 'stuff' 类别
        thing_pq, thing_sq, thing_rq = 0.0, 0.0, 0.0
        stuff_pq, stuff_sq, stuff_rq = 0.0, 0.0, 0.0
        thing_count, stuff_count = 0, 0
        
        for cat, metrics in per_class.items():
            if cat in self.thing_ids:
                thing_pq += metrics['pq']
                thing_sq += metrics['sq']
                thing_rq += metrics['rq']
                thing_count += 1
            else:
                stuff_pq += metrics['pq']
                stuff_sq += metrics['sq']
                stuff_rq += metrics['rq']
                stuff_count += 1
                
        # 计算平均值
        if thing_count > 0:
            thing_pq /= thing_count
            thing_sq /= thing_count
            thing_rq /= thing_count
            
        if stuff_count > 0:
            stuff_pq /= stuff_count
            stuff_sq /= stuff_count
            stuff_rq /= stuff_count
        
        return {
            'pq': pq*100, 'sq': sq*100, 'rq': rq*100,
            'tp': self.TP, 'fp': self.FP, 'fn': self.FN,
            'iou_sum': self.iou_sum,
            'per_class': per_class,
            'thing_pq': thing_pq, 'thing_sq': thing_sq, 'thing_rq': thing_rq,
            'stuff_pq': stuff_pq, 'stuff_sq': stuff_sq, 'stuff_rq': stuff_rq
        }


def compute_cityscapes_pq(panoptic_pred, gt_semantic, gt_instance=None, thing_ids=None, count_th=100, mask_th=0.5, overlap_th=0.5, max_ins=32000):
    """
    计算 Cityscapes 的 Panoptic Quality 指标
    
    Parameters
    ----------
    panoptic_pred : numpy.ndarray
        预测的全景分割结果, shape=(H,W), 包含从0到N-1的整数ID
    gt_semantic : numpy.ndarray
        真实的语义分割标签图, shape=(H,W)
    gt_instance : numpy.ndarray 或 None
        真实的实例分割标签图, shape=(H,W), 可选参数
    thing_ids : set 或 None
        需要区分实例的类别ID集合，默认为 Cityscapes 的 things 类别
    count_th : int
        实例大小阈值，小于此值的实例将被忽略
    mask_th : float
        掩码阈值，小于此值的像素将被忽略
    overlap_th : float
        重叠阈值，预测实例与真实实例重叠比例小于此值的将被忽略
    max_ins : int
        实例ID的最大值
        
    Returns
    -------
    dict
        包含 'pq', 'sq', 'rq' 等评估指标的字典
    """
    # 默认 Cityscapes things 类别
    if thing_ids is None:
        thing_ids = {11, 12, 13, 14, 15, 16, 17, 18}  # person, rider, car, truck, bus, train, motorcycle, bicycle
    
    # 移除小区域
    cleaned_pred = panoptic_pred.copy()
    for seg_id, count in zip(*np.unique(panoptic_pred, return_counts=True)):
        if count < count_th:
            cleaned_pred[panoptic_pred == seg_id] = 0
    
    # 创建评估器并计算指标
    evaluator = CityscapesPanopticEvaluator(thing_ids=thing_ids)
    evaluator.add_image(cleaned_pred, gt_semantic, gt_instance)
    
    return evaluator.evaluate() 