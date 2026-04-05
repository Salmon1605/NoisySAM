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

    def _calculate_IoU(self, groundtruth_mask, predicted_mask): 
        groundtruth_mask = np.int8(groundtruth_mask) 
        predicted_mask = np.int8(predicted_mask) 

        intersection = np.logical_and(groundtruth_mask, predicted_mask).sum() 
        union = np.logical_or(groundtruth_mask, predicted_mask).sum() 
        IoU = (intersection / (union + self.eps)) * 100
        return IoU 
    
    def _calculate_Dice(self, groundtruth_mask, predicted_mask): 
        groundtruth_mask = np.int8(groundtruth_mask) 
        predicted_mask = np.int8(predicted_mask) 

        intersection = (np.logical_and(groundtruth_mask, predicted_mask)).sum() 
        denominator = (groundtruth_mask.sum()) + (predicted_mask.sum()) + self.eps 
        Dice = (intersection / denominator) * 100 
        return Dice
    
    def _boundary_pts(self, mask):
        return np.argwhere(mask ^ ndimage.binary_erosion(mask))
    
    def _compute_hausdorff_95(self, groundtruth_mask:np.ndarray, predicted_mask:np.ndarray) -> float:
        p, g = predicted_mask.astype(bool), groundtruth_mask.astype(bool)
        diag = float(np.sqrt(p.shape[0]**2 + p.shape[1]**2))
        if p.sum() == 0 and g.sum() == 0: return 0.0
        if p.sum() == 0 or g.sum() == 0: return diag
        pp, gp = self._boundary_pts(p), self._boundary_pts(g)
        if len(pp) == 0 or len(gp) == 0: return diag
        tree_g = cKDTree(gp)
        tree_p = cKDTree(pp)
        d1, _ = tree_g.query(pp)
        d2, _ = tree_p.query(gp)
        return float(np.percentile(np.concatenate([d1, d2]), 95))

        