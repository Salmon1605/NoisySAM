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
from utils.dataLoader import COCOLoader, VOCPascalLoader, ADE20KLoader 
from utils.metrics import Metrics 
from utils.noise_injection import NoiseInjection
from utils.run import Experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

OUTPUT_DIR = r'C:\Users\ADMIN\Documents\SAM\Research\result'

# ===== DATA INITIALIZATION =======
 
# COCO dataset
coco_datapath = r'C:\\Users\\ADMIN\\fiftyone\\coco-2017\\validation\\data'
coco_labelpath = r'C:\Users\ADMIN\fiftyone\coco-2017\raw\instances_val2017.json' 
coco_dataset = COCOLoader(coco_datapath=coco_datapath, coco_labelpath=coco_labelpath)

# VOC dataset 
voc_dataset = VOCPascalLoader(split='val')

# ADE20K dataset 
ade_dataset = ADE20KLoader(split='validation') 


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
    "poisson": noise._inject_poisson_noise,
    "salt_pepper": noise._inject_salt_and_pepper_noise,
    "speckle": noise._inject_speckle_noise,
    "pixelate": noise._inject_pixelate,
    "defocus": noise._inject_defocus_noise, 
    "motion_blur": noise._motion_blur
}

# ====== METRICS ====== 
metrics = Metrics() 


# ====== COCO EVALUATION ======= 
# 1 - SAM_b 
COCO_vit_b_configs = {
    "dataset_name": "COCO",
    "model_name": "SAM_b",
    "sam_checkpoint": "C:\Users\ADMIN\Documents\SAM\Research\model\sam_vit_b_01ec64.pth",
    "severities": [3],
    "corruption_types":list(noise_dict.keys()),
    "prompt_type": "bbox",
    "metrics": ["dice", "iou"],
    "primary_metric": "iou",
    "device": "cuda",
    "seed": 42,
    "output_dir": OUTPUT_DIR,
    "experiment_tag": "SAM_vit_b_coco"
}
coco_vit_b_experiment = Experiment(
    model=sam_b_model, 
    dataset=coco_dataset, 
    noise_dict=noise_dict, 
    severity=3
)
coco_vit_b_result = coco_vit_b_experiment._evaluate() 
coco_vit_b_experiment._save_json(configurations=COCO_vit_b_configs)


# 2 - SAM_h 
COCO_vit_h_configs = {
    "dataset_name": "COCO",
    "model_name": "SAM_h",
    "sam_checkpoint": sam_vit_h_checkpoint,
    "severities": [3],
    "corruption_types": list(noise_dict.keys()),
    "prompt_type": "bbox",
    "metrics": ["dice", "iou"],
    "primary_metric": "iou",
    "device": "cuda",
    "seed": 42,
    "output_dir": OUTPUT_DIR,
    "experiment_tag": "SAM_vit_h_coco"
}
coco_vit_h_experiment = Experiment(
    model=sam_h_model, 
    dataset=coco_dataset, 
    noise_dict=noise_dict, 
    severity=3
)
coco_vit_h_result = coco_vit_h_experiment._evaluate() 
coco_vit_h_experiment._save_json(configurations=COCO_vit_h_configs)

# 3 - MobileSAM 
COCO_mobileSAM_configs = {
    "dataset_name": "COCO",
    "model_name": "MobileSAM_t",
    "checkpoint": mobileSAM_checkpoint,
    "severities": [3],
    "corruption_types": list(noise_dict.keys()),
    "prompt_type": "bbox",
    "metrics": ["dice", "iou"],
    "output_dir": OUTPUT_DIR,
    "experiment_tag": "MobileSAM_t_coco"
}
coco_mobile_experiment = Experiment(
    model=mobileSAM_model, 
    dataset=coco_dataset, 
    noise_dict=noise_dict, 
    severity=3
)
coco_mobile_result = coco_mobile_experiment._evaluate() 
coco_mobile_experiment._save_json(configurations=COCO_mobileSAM_configs)

# 4 - FastSAM
COCO_fastSAM_configs = {
    "dataset_name": "COCO",
    "model_name": "FastSAM_x",
    "checkpoint": FastSAM_model, 
    "severities": [3],
    "corruption_types": list(noise_dict.keys()),
    "prompt_type": "bbox",
    "metrics": ["dice", "iou"],
    "output_dir": OUTPUT_DIR,
    "experiment_tag": "FastSAM_coco"
}
coco_fast_experiment = Experiment(
    model=fastSAM_model, 
    dataset=coco_dataset, 
    noise_dict=noise_dict, 
    severity=3
)
coco_fast_result = coco_fast_experiment._evaluate() 
coco_fast_experiment._save_json(configurations=COCO_fastSAM_configs)



# ====== VOC EVALUATION =======
voc_models = [
    (sam_b_model, "SAM_b", sam_vit_b_checkpoint),
    (sam_h_model, "SAM_h", sam_vit_h_checkpoint),
    (mobileSAM_model, "MobileSAM_t", mobileSAM_checkpoint),
    (fastSAM_model, "FastSAM", FastSAM_model)
]

for m_obj, m_name, m_ckpt in voc_models:
    logging.info(f"Running VOC Evaluation for {m_name}...")
    
    voc_configs = {
        "dataset_name": "VOC_Pascal",
        "model_name": m_name,
        "checkpoint": m_ckpt,
        "severities": [3],
        "corruption_types": list(noise_dict.keys()),
        "prompt_type": "bbox",
        "metrics": ["dice", "iou"],
        "output_dir": OUTPUT_DIR,
        "experiment_tag": f"{m_name}_voc"
    }
    
    voc_experiment = Experiment(
        model=m_obj, 
        dataset=voc_dataset, 
        noise_dict=noise_dict, 
        severity=3
    )
    voc_result = voc_experiment._evaluate() 
    voc_experiment._save_json(configurations=voc_configs)

# ====== ADE20k EVALUATION =======  

ade_models = [
    (sam_b_model, "SAM_b", sam_vit_b_checkpoint),
    (sam_h_model, "SAM_h", sam_vit_h_checkpoint),
    (mobileSAM_model, "MobileSAM_t", mobileSAM_checkpoint),
    (fastSAM_model, "FastSAM", FastSAM_model)
]

for m_obj, m_name, m_ckpt in ade_models:
    logging.info(f"Running ADE20k Evaluation for {m_name}...")
    
    ade_configs = {
        "dataset_name": "ADE20K",
        "model_name": m_name,
        "checkpoint": m_ckpt,
        "severities": [3],
        "corruption_types": list(noise_dict.keys()),
        "prompt_type": "bbox",
        "metrics": ["dice", "iou"],
        "output_dir": OUTPUT_DIR,
        "experiment_tag": f"{m_name}_ade20k"
    }
    
    ade_experiment = Experiment(
        model=m_obj, 
        dataset=ade_dataset, 
        noise_dict=noise_dict, 
        severity=3
    )
    ade_result = ade_experiment._evaluate() 
    ade_experiment._save_json(configurations=ade_configs)