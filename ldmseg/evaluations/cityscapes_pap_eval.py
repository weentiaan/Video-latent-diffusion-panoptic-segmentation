"""
Author: (Adapted from KITTI evaluator)
Cityscapes panoptic evaluator
"""

import numpy as np
from scipy import ndimage

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

    def add_image(self, pred_seg, gt_semseg):
        """
        Parameters
        ----------
        pred_seg      : (H,W) 预测的全景分割结果（实例分割）
        gt_semseg     : (H,W) 真值语义分割标签
        """
        # 将-1和ignore_label设为相同的值
        pred_seg = pred_seg.copy()
        pred_seg[pred_seg == -1] = self.ignore_label
        
        # 预处理真值：为thing类别生成实例ID
        gt_instance = np.zeros_like(gt_semseg)
        for thing_id in self.thing_ids:
            thing_mask = gt_semseg == thing_id
            if thing_mask.any():
                # 为每个连通区域分配唯一ID
                labeled, num_components = ndimage.label(thing_mask)
                if num_components > 0:
                    # 只在mask区域设置标签
                    gt_instance[thing_mask] = labeled[thing_mask]

        # 构建GT全景分割
        gt_pan = self._to_panoptic(gt_semseg, gt_instance)
        
        # 对预测分割进行处理：将不同的实例ID映射到唯一的全景ID
        pred_pan = np.zeros_like(pred_seg)
        for i, label in enumerate(np.unique(pred_seg)):
            if label == self.ignore_label:
                continue
                
            # 检查label的语义类别
            if label in self.thing_ids:  # 对于thing类别，保持独立实例
                instance_mask = pred_seg == label
                components, num_components = ndimage.label(instance_mask)
                
                for j in range(1, num_components + 1):
                    component_mask = components == j
                    # 为每个连通区域分配唯一的panoptic ID
                    pred_pan[component_mask] = label * self.max_ins + j
            else:  # 对于stuff类别，直接使用语义标签
                pred_pan[pred_seg == label] = label
        
        # 忽略区域不参与评估
        pred_pan[gt_semseg == self.ignore_label] = -1
        pred_pan[pred_seg == self.ignore_label] = -1
        gt_pan[gt_semseg == self.ignore_label] = -1
        
        # 获取所有有效的GT和预测区域ID
        gt_ids = np.unique(gt_pan)
        pred_ids = np.unique(pred_pan)
        gt_ids = gt_ids[gt_ids != -1]
        pred_ids = pred_ids[pred_ids != -1]
        
        # 创建区域掩码
        gt_masks = {gid: gt_pan == gid for gid in gt_ids}
        pred_masks = {pid: pred_pan == pid for pid in pred_ids}
        
        # 匹配过程
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
                
            # 寻找最佳匹配
            best_iou, best_pid = 0.0, None
            for pid, pmask in pred_masks.items():
                pcat = (pid // self.max_ins) if pid >= self.max_ins else pid
                
                # 类别必须一致才能匹配
                if pcat != gcat:
                    continue
                    
                iou = self._iou(gmask, pmask)
                if iou > best_iou:
                    best_iou, best_pid = iou, pid
            
            # 如果IoU超过阈值，则认为是成功匹配
            if best_iou >= self.iou_thresh:
                self.TP += 1
                self.iou_sum += best_iou
                matched_pred.add(best_pid)
                
                # 更新类别级别统计
                self.TP_per_class[gcat] += 1
                self.iou_sum_per_class[gcat] += best_iou
            else:
                # 未匹配的GT区域计为FN
                self.FN += 1
                if gcat in self.FN_per_class:
                    self.FN_per_class[gcat] += 1
                else:
                    self.FN_per_class[gcat] = 1
        
        # 未匹配的预测区域计为FP
        self.FP += len(pred_ids) - len(matched_pred)
        
        # 更新类别级别的FP
        for pid in pred_ids:
            if pid not in matched_pred:
                pcat = (pid // self.max_ins) if pid >= self.max_ins else pid
                if pcat not in self.FP_per_class:
                    self.FP_per_class[pcat] = 0
                self.FP_per_class[pcat] += 1

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
                denom = tp + 0.5 * (fp + fn)
                cat_rq = tp / denom if denom > 0 else 0.0
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
            'thing_pq': thing_pq*100, 'thing_sq': thing_sq*100, 'thing_rq': thing_rq*100,
            'stuff_pq': stuff_pq*100, 'stuff_sq': stuff_sq*100, 'stuff_rq': stuff_rq*100
        }


def compute_cityscapes_pq(panoptic_pred, gt_semantic, thing_ids=None, count_th=100, mask_th=0.5, overlap_th=0.5, max_ins=32000):
    """
    计算 Cityscapes 的 Panoptic Quality 指标
    
    Parameters
    ----------
    panoptic_pred : numpy.ndarray
        预测的全景分割结果, shape=(H,W), 包含从0到N-1的整数ID
    gt_semantic : numpy.ndarray
        真实的语义分割标签图, shape=(H,W)
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
    evaluator.add_image(cleaned_pred, gt_semantic)
    
    return evaluator.evaluate() 