import os 
import sys
import re 
import cv2  
import tqdm 
import logging
import json
import glob
import tqdm
import warnings

import torch 
import torchvision 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle as pkl

from utils.modelLoader import SAM1, MobileSAM, FastSAMModel 
from utils.dataLoader import COCOLoader, VOCPascalLoader, ADE20KLoader, BSDS500Loader
from utils.metrics import Metrics 
from utils.noise_injection import NoiseInjection
from utils.run import Experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

OUTPUT_DIR = r'C:\Users\ADMIN\Documents\SAM\Research\result' 
THRESHOLD = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== DATA INITIALIZATION =======
 
# COCO dataset
coco_datapath = r'C:\\Users\\ADMIN\\fiftyone\\coco-2017\\validation\\data'
coco_labelpath = r'C:\Users\ADMIN\fiftyone\coco-2017\raw\instances_val2017.json' 
coco_dataset = COCOLoader(coco_datapath=coco_datapath, coco_labelpath=coco_labelpath)

# VOC dataset 
voc_dataset = VOCPascalLoader(split='val')

# ADE20K dataset 
ade_dataset = ADE20KLoader(split='validation') 

# BSDS500 dataset
root = r'C:\Users\ADMIN\Documents\SAM\Research\src\Data\BSR_bsds500\BSR\BSDS500\data' 
images_subdir = r'C:\Users\ADMIN\Documents\SAM\Research\src\Data\BSR_bsds500\BSR\BSDS500\data\images' 
annoations_subdir = r'C:\Users\ADMIN\Documents\SAM\Research\src\Data\BSR_bsds500\BSR\BSDS500\data\groundTruth' 
split = 'val' 

bsds500_dataset = BSDS500Loader(image_subdir=images_subdir, annotations_subdir=annoations_subdir, root=root, split=split) 


# ====== MODEL INITIALIZATION ======= 

# SAM 1  
sam_vit_b_checkpoint = r'C:\Users\ADMIN\Documents\SAM\Research\model\sam_vit_b_01ec64.pth'
sam_vit_h_checkpoint = r'C:\Users\ADMIN\Documents\SAM\Research\model\sam_vit_h_4b8939.pth'

sam_configurations = [ 
    {'model_type':'vit_b', 'checkpoint':sam_vit_b_checkpoint}, 
    {'model_type':'vit_h', 'checkpoint':sam_vit_h_checkpoint}
]
sam_b_model = SAM1(model_name='SAM1_b',
                model_type=sam_configurations[0]['model_type'],
                checkpoint= sam_configurations[0]['checkpoint']
                )

sam_h_model = SAM1(model_name='SAM1_h', 
                   model_type=sam_configurations[1]['model_type'],
                   checkpoint=sam_configurations[1]['checkpoint']
                ) 

# MobileSAM
mobileSAM_checkpoint = r'C:\Users\ADMIN\Documents\SAM\Research\model\mobile_sam.pt' 
mobileSAM_configurations = [ 
    {'model_type':'vit_t', 'checkpoint':mobileSAM_checkpoint}
] 

mobileSAM_model = MobileSAM(model_name='MobileSAM_t',
                            model_type=mobileSAM_configurations[0]['model_type'],
                            checkpoint=mobileSAM_configurations[0]['checkpoint']
                            )

# FastSAM 
FastSAM_model = r'C:\Users\ADMIN\Documents\SAM\Research\model\FastSAM-x.pt' 
fastSAM_configurations = [ 
    {'model':FastSAM_model}
]

fastSAM_model = FastSAMModel(model_name='FastSAM', 
                            model=fastSAM_configurations[0]['model']
                            )


# ====== NOISE ====== 
noise = NoiseInjection() 

noise_dict = {
    "gaussian": noise._inject_gaussian_noise,
    "motion_blur": noise._motion_blur,
    "snow": noise._inject_snow, 
    "brightness": noise._inject_brightness, 
    "contrast": noise._inject_contrast, 
    "jpeg": noise._inject_JPEG
}

severities = [1, 3, 5]

# ====== METRICS ====== 
metrics = Metrics() 



# ====== COCO EVALUATION ======
coco_models = [ 
    (sam_b_model, "SAM_b", sam_vit_b_checkpoint),
    (sam_h_model, "SAM_h", sam_vit_h_checkpoint),
    (mobileSAM_model, "MobileSAM", mobileSAM_checkpoint),
    (fastSAM_model, "FastSAM", FastSAM_model)
]

for m_obj, m_name, m_ckpt in coco_models:
    logging.info(f"Running COCO Evaluation for {m_name}, model_type: {m_name}, checkpoint: {m_ckpt}")
    
    coco_configs = {
        "dataset_name": "COCO",
        "model_name": m_name,
        "checkpoint": m_ckpt,
        "severities": severities,
        "corruption_types": list(noise_dict.keys()),
        "prompt_type": "bbox",
        "metrics": ["dice", "iou", "hd95"],
        "output_dir": OUTPUT_DIR,
        "experiment_tag": f"{m_name}_coco"
    }
    
    coco_experiment = Experiment(
        model=m_obj, 
        dataset=coco_dataset, 
        noise_dict=noise_dict, 
        severities=severities
    )
    coco_result, coco_failure_cases = coco_experiment._evaluate() 
    coco_experiment._save_json(configurations=coco_configs)

# ====== VOC EVALUATION =======
voc_models = [
    (sam_b_model, "SAM_b", sam_vit_b_checkpoint),
    (sam_h_model, "SAM_h", sam_vit_h_checkpoint),
    (mobileSAM_model, "MobileSAM", mobileSAM_checkpoint),
    (fastSAM_model, "FastSAM", FastSAM_model)
]

for m_obj, m_name, m_ckpt in voc_models:
    logging.info(f"Running VOC Evaluation for {m_name}, model_type: {m_name}, checkpoint: {m_ckpt} ")
    
    voc_configs = {
        "dataset_name": "VOC_Pascal",
        "model_name": m_name,
        "checkpoint": m_ckpt,
        "severities": severities,
        "corruption_types": list(noise_dict.keys()),
        "prompt_type": "bbox",
        "metrics": ["dice", "iou", "hd95"],
        "output_dir": OUTPUT_DIR,
        "experiment_tag": f"{m_name}_voc"
    }
    
    voc_experiment = Experiment(
        model=m_obj, 
        dataset=voc_dataset, 
        noise_dict=noise_dict, 
        severities=severities
    )
    voc_result, voc_failure_cases = voc_experiment._evaluate() 
    voc_experiment._save_json(configurations=voc_configs)

# ====== ADE20k EVALUATION =======  

ade_models = [
    (sam_b_model, "SAM_b", sam_vit_b_checkpoint),
    (sam_h_model, "SAM_h", sam_vit_h_checkpoint),
    (mobileSAM_model, "MobileSAM_t", mobileSAM_checkpoint),
    (fastSAM_model, "FastSAM", FastSAM_model)
]

for m_obj, m_name, m_ckpt in ade_models:
    logging.info(f"Running ADE20k Evaluation for {m_name}, model_type: {m_name}, checkpoint: {m_ckpt}")
    
    ade_configs = {
        "dataset_name": "ADE20K",
        "model_name": m_name,
        "checkpoint": m_ckpt,
        "severities": severities,
        "corruption_types": list(noise_dict.keys()),
        "prompt_type": "bbox",
        "metrics": ["dice", "iou", "hd95"],
        "output_dir": OUTPUT_DIR,
        "experiment_tag": f"{m_name}_ade20k"
    }
    
    ade_experiment = Experiment(
        model=m_obj, 
        dataset=ade_dataset, 
        noise_dict=noise_dict, 
        severities=severities
    )
    ade_result, ade_failure_cases = ade_experiment._evaluate() 
    ade_experiment._save_json(configurations=ade_configs)

# ======  EVALUATION BSDS500======= 

bsds_models = [
    (sam_b_model, "SAM_b", sam_vit_b_checkpoint),
    (sam_h_model, "SAM_h", sam_vit_h_checkpoint),
    (mobileSAM_model, "MobileSAM_t", mobileSAM_checkpoint),
    (fastSAM_model, "FastSAM", FastSAM_model)
]

for m_obj, m_name, m_ckpt in bsds_models: 
    logging.info(f"Running BSDS500 Evaluation for {m_name}, model_type: {m_name}, checkpoint: {m_ckpt}") 

    bsds_configs = { 
        "dataset_name": "BSDS500",
        "model_name": m_name,
        "checkpoint": m_ckpt,
        "severities": severities,
        "corruption_types": list(noise_dict.keys()),
        "prompt_type": "bbox",
        "metrics": ["dice", "iou", "hd95"],
        "output_dir": OUTPUT_DIR,
        "experiment_tag": f"{m_name}_bsds500"
    }

    bsds_experiment = Experiment(
        model=m_obj, 
        dataset=bsds500_dataset, 
        noise_dict=noise_dict, 
        severities=severities 
    )

    bsds_result, bsds_failure_cases = bsds_experiment._evaluate() 
    bsds_experiment._save_json(configurations=bsds_configs)
