import os 
import sys 
import re 
import glob 
import tqdm
import json
import logging

import numpy as np 
import matplotlib as plt 
from utils.metrics import Metrics 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

metrics = Metrics() 

class Experiment:
    def __init__(self, model, dataset, noise_dict:dict, severities:list):
        self.model = model 
        self.dataset = dataset 
        self.noise_dict = noise_dict 
        self.severities = severities 
        self.result = []
        self.failure = []

    def _is_failure(self, pred_mask): 
        if pred_mask is None:
            return True
        if isinstance(pred_mask, np.ndarray) and pred_mask.sum() == 0:
            return True
        return False

    def _evaluate_each_noise(self, noise_name, noise_fn, severity): 
        noise_result = { 
            'corruption':noise_name, 
            'severity':severity, 
            'images': [], 
            'mean_IoU': None, 
            'mean_DICE': None, 
            'mean_HD95': None,
        }
        failure_cases = { 
            'corruption':noise_name, 
            'severity':severity, 
            'images': [],
        }

        total_IoU = [] 
        total_DICE = [] 
        total_HD95 = []

        for sample in tqdm.tqdm(self.dataset, desc=f"Noise: {noise_name} | severity: {severity} "): 
            image = sample['image']
            transformed_image = noise_fn(image, severity=severity) 
            self.model._set_image(image=transformed_image) 
            sample_IoU = []
            sample_Dice = []
            sample_HD95 = []
            image_result = {
                'image_id': str(sample['image_id']), 
                'annotations': []
            }

            for ann_id, (bbox, gt_mask, label) in enumerate(
                zip(sample["bounding_boxes"], sample["masks"], sample["labels"])
            ):
                
                input_bbox = np.array(bbox) 
                output = self.model._predict(input_bbox) 
                pred_mask = output['mask']
                scores = output['score'] 

                if self._is_failure(pred_mask=pred_mask): 
                    failure_cases['images'].append({
                        "image_id": str(sample["image_id"]),
                        "ann_id": int(ann_id),
                        "bbox": bbox,
                        "category": str(label),
                        "status": str(output['status']),
                        "reason": str(output['reason'])
                    })
                    continue

                iou = metrics._calculate_IoU(gt_mask, pred_mask) 
                dice = metrics._calculate_Dice(gt_mask, pred_mask) 
                hd95 = metrics._compute_hausdorff_95(gt_mask, pred_mask)

                sample_IoU.append(iou)
                sample_Dice.append(dice) 
                sample_HD95.append(hd95)

                ann_block = {
                    "ann_id": int(ann_id),
                    "image_id": str(sample["image_id"]),
                    "category_name": str(label),
                    "iou_pred": float(scores[0]),
                    "dice": float(dice),
                    "iou": float(iou),
                    "hd95": float(hd95)
                }

                image_result["annotations"].append(ann_block)

            noise_result["images"].append(image_result)

            if len(sample_IoU) > 0:
                total_IoU.append(np.mean(sample_IoU))
                total_DICE.append(np.mean(sample_Dice))
                total_HD95.append(np.mean(sample_HD95))
        
        noise_result['mean_IoU'] = float(np.mean(total_IoU)) if total_IoU else 0.0 
        noise_result['mean_DICE'] = float(np.mean(total_DICE)) if total_DICE else 0.0 
        noise_result['mean_HD95'] = float(np.mean(total_HD95)) if total_HD95 else 0.0

        return noise_result, failure_cases


    def _evaluate(self): 
        for severity in self.severities:
            for noise_name, noise_fn in self.noise_dict.items(): 
                result, failure_cases = self._evaluate_each_noise(noise_name=noise_name, noise_fn=noise_fn, severity=severity)
                self.result.append(result) 
                self.failure.append(failure_cases)
        
        final_result = self.result 
        final_failure = self.failure
        return final_result, final_failure
    
    def _save_json(self, configurations): 
        final_output = { 
            "configurations": configurations, 
            "results": self.result, 
            "failure_cases": self.failure
        }
        os.makedirs(configurations["output_dir"], exist_ok=True) 
        output_path = os.path.join(
            configurations['output_dir'], 
            f"{configurations['experiment_tag']}_result.json"
        )
        with open(output_path, "w") as f: 
            json.dump(final_output, f, indent=4) 

        logging.info(f"Saved to {output_path}")