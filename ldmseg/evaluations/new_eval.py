import numpy as np
import six

# 固定参数：instance 部分用6位编码，即MAX_INS = 2**6 = 64
MAX_INS = 64
# 对于 KITTI，我们认为评价时有效类别为：
# [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
VALID_CAT_IDS = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
NUM_EVAL_CAT = len(VALID_CAT_IDS)  # 20个类别
# 构造连续映射：dataset类别 id -> 连续索引（0 ~ 19）
EVAL_MAPPING = {cat_id: idx for idx, cat_id in enumerate(VALID_CAT_IDS)}

def vpq_eval(element, num_cat=NUM_EVAL_CAT):
    """
    计算单张图像的 VPQ 统计信息，并返回 (iou_per_class, tp_per_class, fn_per_class, fp_per_class)。
    
    输入 element 为 [pred, gt]，其中 pred 和 gt 均为 numpy 数组，形状 (H, W)。
    编码规则假定为：每个像素值 = category * MAX_INS + instance，
    其中 category 为 KITTI 中的原始类别 id（例如 10, 11, ..., 29），
    而 instance 为实例编码（取值范围 0 ~ MAX_INS-1）。
    
    只有当预测和 GT 的 category 都在 VALID_CAT_IDS 中时，才会计入统计；
    否则，跳过（例如 unlabeled/outlier）。
    """
    offset = 2 ** 30  # 用于构造联合编码
    # 初始化统计数组，长度固定为 num_cat（20）
    iou_per_class = np.zeros(num_cat, dtype=np.float64)
    tp_per_class = np.zeros(num_cat, dtype=np.float64)
    fn_per_class = np.zeros(num_cat, dtype=np.float64)
    fp_per_class = np.zeros(num_cat, dtype=np.float64)

    def _ids_to_counts(arr):
        ids, counts = np.unique(arr, return_counts=True)
        return dict(zip(ids, counts))
    
    pred_areas = _ids_to_counts(element[0])
    gt_areas = _ids_to_counts(element[1])
    
    # 此处暂不做对 void 的特殊处理（可根据需要扩展 ignore 规则）
    
    # 构造联合编码：这样可以方便计算交集
    int_ids = element[1].astype(np.int64) * offset + element[0].astype(np.int64)
    int_areas = _ids_to_counts(int_ids)
    
    def prediction_void_overlap(pred_id):
        # 可扩展：对 void 类区域 overlap 的处理（目前直接返回0）
        return 0

    def prediction_ignored_overlap(pred_id):
        # 此处略过 ignored overlap 处理
        return 0

    gt_matched = set()
    pred_matched = set()

    for int_id, area in six.iteritems(int_areas):
        gt_id = int(int_id // offset)
        pred_id = int(int_id % offset)
        gt_cat = int(gt_id // MAX_INS)
        pred_cat = int(pred_id // MAX_INS)
        # 如果预测和 GT 的类别不在有效集合中，则跳过
        if gt_cat not in EVAL_MAPPING or pred_cat not in EVAL_MAPPING:
            continue
        # 使用连续映射后的类别
        cont_gt = EVAL_MAPPING[gt_cat]
        cont_pred = EVAL_MAPPING[pred_cat]
        # 如果两者不同，则不认为匹配
        if cont_gt != cont_pred:
            continue
        union = gt_areas.get(gt_id, 0) + pred_areas.get(pred_id, 0) - area - prediction_void_overlap(pred_id)
        iou = area / union if union > 0 else 0
        if iou > 0.5:
            tp_per_class[cont_gt] += 1
            iou_per_class[cont_gt] += iou
            gt_matched.add(gt_id)
            pred_matched.add(pred_id)
    
    for gt_id, area in gt_areas.items():
        cat = gt_id // MAX_INS
        if cat not in EVAL_MAPPING:
            continue
        if gt_id in gt_matched:
            continue
        cont_idx = EVAL_MAPPING[cat]
        fn_per_class[cont_idx] += 1

    for pred_id, area in pred_areas.items():
        if pred_id in pred_matched:
            continue
        cat = pred_id // MAX_INS
        if cat not in EVAL_MAPPING:
            continue
        cont_idx = EVAL_MAPPING[cat]
        fp_per_class[cont_idx] += 1

    return (iou_per_class, tp_per_class, fn_per_class, fp_per_class)