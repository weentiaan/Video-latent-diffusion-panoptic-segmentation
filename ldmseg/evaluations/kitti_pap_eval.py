#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panoptic evaluation & visualisation for a custom setting

Inputs per image
----------------
pred : H × W  (np.int32 / torch.LongTensor)        # 模型输出（0~30，共 31 个 panoptic 类）
semseg : H × W (np.int32 / torch.LongTensor)       # 语义 GT，像素值与 pred 同语义空间
instance : H × W (np.int32 / torch.LongTensor)     # 实例 GT，0 表示 background/void，其余值为实例 id
image_semseg : H × W × 3 (np.uint8)                # 可选，三通道 RGB 的 panoptic 伪彩图（固定调色板）

Author: ChatGPT – 2025‑04‑21
License: MIT
"""

# ----------------------------------------------------------------------
# 依赖
# ----------------------------------------------------------------------
from __future__ import annotations

import os
import colorsys
import random
from pathlib import Path
from typing import Dict, Tuple, List, Set

import numpy as np
from PIL import Image
from tqdm import tqdm

# 如果你只做 CPU 运算，可以把 torch 相关行注释掉
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 调色板工具
# ----------------------------------------------------------------------
def get_random_color(seed: int) -> Tuple[int, int, int]:
    random.seed(seed)
    h = random.random()
    s = 0.6 + random.random() * 0.4
    v = 0.7 + random.random() * 0.3
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def build_palette(num_semantic_cls: int, thing_ids: Set[int], max_instances: int) -> Dict[int, Tuple[int, int, int]]:
    """
    为 0~num_semantic_cls 以及实例 id -> 颜色  建一个彩色表
    panoptic id = sem_id * max_instances + inst_id
    stuff（非 thing）始终用 inst_id = 0
    """
    palette: Dict[int, Tuple[int, int, int]] = {}
    for sem in range(num_semantic_cls):
        base_color = get_random_color(sem + 13)  # 固定可重现
        if sem in thing_ids:
            for ins in range(max_instances):
                palette[sem * max_instances + ins] = base_color
        else:
            palette[sem * max_instances + 0] = base_color
    palette[-1] = (0, 0, 0)  # void / ignore
    return palette


# ----------------------------------------------------------------------
# Panoptic Evaluator
# ----------------------------------------------------------------------
class KITTIPanopticEvaluator:
    """
    简化版 PQ/SQ/RQ 评价器
    pred, semseg, instance 均是 numpy/torch Tensor 均可
    thing_ids : 需要区分实例的类 id（你的 0‑30 里任选）
    max_ins   : 给一个很大的整数即可，如 2**20
    """

    def __init__(self, thing_ids: Set[int], num_semantic_cls: int, max_ins: int = 1 << 20,
                 iou_thr: float = .5, ignore_label: int = 0) -> None:
        self.thing_ids = {10, 11, 12, 13, 14, 15, 16, 17}
        self.num_semantic_cls = 30
        self.max_ins = 50
        self.iou_thr = 0.5
        self.ignore_label = 0
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.pq_stat = dict(tp=0, fp=0, fn=0, iou=0.0)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    # ------------------------------------------------------------------
    def _combine_gt(self, semseg: np.ndarray, instance: np.ndarray) -> np.ndarray:
        """semseg + instance -> panoptic id"""
        semseg = semseg.astype(np.int64)
        instance = instance.astype(np.int64)
        pan_gt = semseg.copy()

        # 对 thing 类别加入 instance
        mask_thing = np.isin(semseg, list(self.thing_ids))
        pan_gt[mask_thing] = semseg[mask_thing] * self.max_ins + instance[mask_thing]

        pan_gt[semseg == self.ignore_label] = -1
        return pan_gt

    # ------------------------------------------------------------------
    @staticmethod
    def _iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return 0.0 if union == 0 else inter / union

    # ------------------------------------------------------------------
    def add_sample(self, pred: np.ndarray, semseg: np.ndarray, instance: np.ndarray):
        """
        pred      : (H, W)  0~num_class-1 （已经是 '类别 id', 不含实例信息）
        semseg    : (H, W)  GT 语义类 id
        instance  : (H, W)  GT instance id（0 表示 void / stuff）
        """
        pred = self._to_numpy(pred).astype(np.int64)
        semseg = self._to_numpy(semseg).astype(np.int64)
        instance = self._to_numpy(instance).astype(np.int64)

        pan_gt = self._combine_gt(semseg, instance)
        # 预测 panoptic（pred 只给类别，这里对 stuff 直接用 inst=0，对 thing 用 inst=1 做个 dummy）
        pan_pred = pred.copy()
        mask_thing = np.isin(pred, list(self.thing_ids))
        pan_pred[mask_thing] = pred[mask_thing] * self.max_ins + 1  # 预测没有分实例 → inst=1
        pan_pred[pred == self.ignore_label] = -1

        gt_ids = np.unique(pan_gt)
        pred_ids = np.unique(pan_pred)
        gt_ids = gt_ids[gt_ids != -1]
        pred_ids = pred_ids[pred_ids != -1]

        gt_masks = {gid: (pan_gt == gid) for gid in gt_ids}
        pred_masks = {pid: (pan_pred == pid) for pid in pred_ids}

        matched_pred: Set[int] = set()
        for gid, gmask in gt_masks.items():
            gcat = gid // self.max_ins if gid >= self.max_ins else gid
            best_iou = 0.0
            best_pid = None
            for pid, pmask in pred_masks.items():
                if pid in matched_pred:
                    continue
                pcat = pid // self.max_ins if pid >= self.max_ins else pid
                if pcat != gcat:
                    continue
                iou = self._iou(gmask, pmask)
                if iou > best_iou:
                    best_iou, best_pid = iou, pid
            if best_iou >= self.iou_thr:
                self.pq_stat['tp'] += 1
                self.pq_stat['iou'] += best_iou
                matched_pred.add(best_pid)
            else:
                self.pq_stat['fn'] += 1
        self.pq_stat['fp'] += len(pred_masks) - len(matched_pred)

    # ------------------------------------------------------------------
    def evaluate(self) -> Dict[str, float]:
        tp, fp, fn, iou = (self.pq_stat[k] for k in ('tp', 'fp', 'fn', 'iou'))
        if tp == 0:
            return dict(PQ=0.0, SQ=0.0, RQ=0.0, TP=tp, FP=fp, FN=fn)

        sq = iou / tp
        rq = tp / (tp + 0.5 * (fp + fn))
        pq = sq * rq
        return dict(PQ=pq, SQ=sq, RQ=rq, TP=tp, FP=fp, FN=fn)


# ----------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------
def save_color_panoptic(mask: np.ndarray, path: str | Path,
                        palette: Dict[int, Tuple[int, int, int]]) -> None:
    """mask (H, W) panoptic id  → RGB"""
    h, w = mask.shape
    img = np.zeros((h, w, 3), np.uint8)
    for pid in np.unique(mask):
        img[mask == pid] = palette.get(pid, (0, 0, 0))
    Image.fromarray(img).save(str(path))


# ----------------------------------------------------------------------
# Demo / 用法
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # ------------------------------------------
    # 假设有批量数据，你可以把下面改成自己的 dataloader
    # ------------------------------------------
    HEIGHT, WIDTH = 256, 512
    N_CLASS = 31
    THING_IDS = {1, 2, 3, 4, 5, 6}           # 举例：0‑30 中哪些类需要区分实例
    evaluator = PanopticEvaluator(thing_ids=THING_IDS, num_semantic_cls=N_CLASS)

    # 随机伪造 10 张图像
    for i in tqdm(range(10)):
        pred = np.random.randint(0, N_CLASS, (HEIGHT, WIDTH), dtype=np.int64)
        semseg = np.random.randint(0, N_CLASS, (HEIGHT, WIDTH), dtype=np.int64)
        instance = np.random.randint(0, 50, (HEIGHT, WIDTH), dtype=np.int64)  # 50 个实例上限

        evaluator.add_sample(pred, semseg, instance)

    res = evaluator.evaluate()
    print(f"PQ={res['PQ']:.4f}, SQ={res['SQ']:.4f}, RQ={res['RQ']:.4f}",
          f" | TP={res['TP']}, FP={res['FP']}, FN={res['FN']}")

    # ---------------- 可视化保存示例 ----------------
    palette = build_palette(num_semantic_cls=N_CLASS,
                            thing_ids=THING_IDS,
                            max_instances=evaluator.max_ins)
    demo_mask = (np.random.rand(HEIGHT, WIDTH) * (N_CLASS - 1)).astype(int)
    save_color_panoptic(demo_mask, "demo_panoptic_rgb.png", palette)
