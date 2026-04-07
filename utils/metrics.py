import os 
import sys 
import re
import glob 
import cv2
import logging  

import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 

from torch.utils.data import Dataset, DataLoader 
from scipy import ndimage
from scipy.spatial import cKDTree

class Metrics: 
    def __init__(self, eps = 1e-6): 
        self.eps = eps 

    def _ensure_binary(self, mask):
        return mask.astype(bool)

    def _calculate_IoU(self, groundtruth_mask, predicted_mask): 
        gt = self._ensure_binary(groundtruth_mask)
        pred = self._ensure_binary(predicted_mask)

        intersection = np.logical_and(gt, pred).sum() 
        union = np.logical_or(gt, pred).sum() 
        return (intersection / (union + self.eps)) * 100
    
    def _calculate_Dice(self, groundtruth_mask, predicted_mask): 
        gt = self._ensure_binary(groundtruth_mask)
        pred = self._ensure_binary(predicted_mask)

        intersection = np.logical_and(gt, pred).sum() 
        denominator = gt.sum() + pred.sum() + self.eps 
        return (2.0 * intersection / denominator) * 100
    
    def _calculate_precision_recall(self, groundtruth_mask, predicted_mask): 
        gt = self._ensure_binary(groundtruth_mask)
        pred = self._ensure_binary(predicted_mask)

        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        return float(precision) * 100, float(recall) * 100

    def _boundary_pts(self, mask):
        return np.argwhere(mask ^ ndimage.binary_erosion(mask))
    
    def _compute_hausdorff_95(self, groundtruth_mask:np.ndarray, predicted_mask:np.ndarray) -> float:
        p, g = self._ensure_binary(predicted_mask), self._ensure_binary(groundtruth_mask)
        diag = float(np.sqrt(p.shape[0]**2 + p.shape[1]**2))
        if not p.any() and not g.any(): return 0.0
        if not p.any() or not g.any(): return diag
        pp, gp = self._boundary_pts(p), self._boundary_pts(g)
        if len(pp) == 0 or len(gp) == 0: return diag
        tree_g = cKDTree(gp)
        tree_p = cKDTree(pp)
        d1, _ = tree_g.query(pp)
        d2, _ = tree_p.query(gp)
        return float(np.percentile(np.concatenate([d1, d2]), 95))
    
        