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

        