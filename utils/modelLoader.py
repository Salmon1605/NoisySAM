import os 
import sys 
import glob 

import torch 
import torchvision 
import torch.nn as nn 
import numpy as np 

from abc import ABC, abstractmethod

from segment_anything import sam_model_registry, SamPredictor 
from mobile_sam import sam_model_registry as mobile_sam_model_registry 
from mobile_sam import SamPredictor as MobileSamPredictor
from ultralytics.models.fastsam import FastSAMPredictor

class AbstractLoader: 
    def __init__(self, model_name):
        self.model_name = model_name 
        # self.model_type = model_type 
        # self.checkpoint = checkpoint
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None 

    @abstractmethod
    def _config(self): 
        pass

    @abstractmethod
    def _predict(self, image, bounding_box):
        pass 

class SAM1(AbstractLoader): 
    def __init__(self, model_name:str=None, model_type:str=None, checkpoint:str=None):
        super().__init__(model_name) 
        self.model_name = model_name 
        self.model_type = model_type 
        self.checkpoint = checkpoint 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.model = self.model.to(self.device) 
        self.predictor = SamPredictor(self.model)



    def _set_image(self, image: np.array): 
        self.predictor.set_image(image=image)
 
    def _predict(self, bounding_box:list): 
        # predictor = SamPredictor(self.model) 
        # predictor.set_image(image if type(image) is np.array else np.array(image)) 

         # bbox must be in [x1, y1, x2, y2] format
        input_bbox = bounding_box if type(bounding_box) is np.array else np.array(bounding_box) 
        masks, scores, logits = self.predictor.predict(
            box=input_bbox[None, :],
            multimask_output=False 
        )
        pred_mask = masks[0]
        return {
            "mask": pred_mask,
            "score": scores,
            "status": "ok",
            "reason": None
        }

class MobileSAM(AbstractLoader): 
    def __init__(self, model_name:str=None, model_type:str=None, checkpoint:str=None):
        super().__init__(model_name) 
        self.model_name = model_name 
        self.model_type = model_type 
        self.checkpoint = checkpoint 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = mobile_sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.model = self.model.to(self.device) 
        self.predictor = MobileSamPredictor(self.model)

    def _set_image(self, image: np.array): 
        self.predictor.set_image(image=image)
 
    def _predict(self, bounding_box:list): 
        # predictor = SamPredictor(self.model) 
        # predictor.set_image(image if type(image) is np.array else np.array(image)) 

         # bbox must be in [x1, y1, x2, y2] format
        input_bbox = bounding_box if type(bounding_box) is np.array else np.array(bounding_box) 
        masks, scores, logits = self.predictor.predict(
            box=input_bbox[None, :],
            multimask_output=False 
        )
        pred_mask = masks[0]
        return {
            "mask": pred_mask,
            "score": scores,
            "status": "ok",
            "reason": None
        }


class FastSAMModel(AbstractLoader): 
    def __init__(
            self,
            model_name, 
            model, 
            task:str='segment',
            conf:float=0.4, 
            mode:str='predict', 
            imgsz:int=512, 
            retina_masks:bool=True):
        super().__init__(model_name)
        self.model_name = model_name 
        device_str = '0' if torch.cuda.is_available() else 'cpu'
        self.overrides = dict(
                        conf=conf, 
                        task=task, 
                        mode=mode, 
                        model=model, 
                        save=False, 
                        imgsz=imgsz, 
                        retina_masks=retina_masks, 
                        device=device_str)  
        self.predictor = FastSAMPredictor(overrides=self.overrides) 

    def _set_image(self, image:np.array):
        self.everything_results = self.predictor(image) 
        
    def _predict(self, bounding_box:np.array): 
        input_bbox = np.array(bounding_box) if bounding_box is not np.array else bounding_box 
        # result = self.predictor.prompt(results=self.everything_results, bboxes=input_bbox)[0] 
       

        # pred_mask = result.masks.data.cpu().numpy() 
        # scores = result.boxes.conf.cpu().numpy()
        # return pred_mask[0], scores

        try: 
            result = self.predictor.prompt(results=self.everything_results, bboxes=input_bbox)[0] 
            if result.masks is None:
                return {
                    "mask": None,
                    "score": None,
                    "status": "fail",
                    "reason": "no_mask"
                }
            pred_mask = result.masks.data.cpu().numpy()[0]
            scores = result.boxes.conf.cpu().numpy()
            return {
                "mask": pred_mask,
                "score": scores,
                "status": "ok",
                "reason": None
             }

        except Exception as e:
            return {
                "mask": None,
                "score": None,
                "status": "fail",
                "reason": str(e)
            }








         
     