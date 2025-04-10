import numpy as np
def vpq_eval(element):
        """
        针对单张图像进行计算。
        element 为 [pred_panoptic, gt_panoptic] 两个 numpy 数组。
        这里采用你提供的 KITTI eval 代码的逻辑。
        """
        max_ins = 64  # 这里采用 64（2^6），即 instance 部分为 6 位
        ign_id = 255
        offset = 2 ** 30  # 这里的 offset 用于区分 gt 和 pred 之间的编码（可保持不变）
        num_cat = 20    # 此处假设类别数设为20（未必会用到全部，这里仅做统计）

        iou_per_class = np.zeros(num_cat, dtype=np.float64)
        tp_per_class = np.zeros(num_cat, dtype=np.float64)
        fn_per_class = np.zeros(num_cat, dtype=np.float64)
        fp_per_class = np.zeros(num_cat, dtype=np.float64)

        def _ids_to_counts(id_array):
            ids, counts = np.unique(id_array, return_counts=True)
            return dict(zip(ids, counts))

        pred_areas = _ids_to_counts(element[0])
        gt_areas = _ids_to_counts(element[1])

        void_id = ign_id * max_ins
        ign_ids = {gt_id for gt_id in gt_areas if (gt_id // max_ins) == ign_id}

        int_ids = element[1].astype(np.int64) * offset + element[0].astype(np.int64)
        int_areas = _ids_to_counts(int_ids)

        def prediction_void_overlap(pred_id):
            void_int_id = void_id * offset + pred_id
            return int_areas.get(void_int_id, 0)

        def prediction_ignored_overlap(pred_id):
            total = 0
            for _ign_id in ign_ids:
                total += int_areas.get(_ign_id * offset + pred_id, 0)
            return total

        gt_matched = set()
        pred_matched = set()

        for int_id, area in int_areas.items():
            gt_id = int(int_id // offset)
            gt_cat = int(gt_id // max_ins)
            pred_id = int(int_id % offset)
            pred_cat = int(pred_id // max_ins)
            if gt_cat != pred_cat:
                continue
            union = gt_areas[gt_id] + pred_areas[pred_id] - area - prediction_void_overlap(pred_id)
            iou = area / union if union > 0 else 0
            if iou > 0.5:
                tp_per_class[gt_cat] += 1
                iou_per_class[gt_cat] += iou
                gt_matched.add(gt_id)
                pred_matched.add(pred_id)

        for gt_id in gt_areas:
            if gt_id in gt_matched:
                continue
            cat_id = gt_id // max_ins
            if cat_id == ign_id:
                continue
            fn_per_class[cat_id] += 1

        for pred_id in pred_areas:
            if pred_id in pred_matched:
                continue
            if (prediction_ignored_overlap(pred_id) / pred_areas[pred_id]) > 0.5:
                continue
            cat = pred_id // max_ins
            fp_per_class[cat] += 1

        return (iou_per_class, tp_per_class, fn_per_class, fp_per_class)