# ======== IMPORT LIBRARIES ============= 

import os 
import sys
import re 
import cv2  
import tqdm 
import logging
import json
import glob
import pickle as pkl 

import torch 
import numpy as np 
import torch.nn as nn 
import torchvision 
import fiftyone as fo
import fiftyone.zoo as foz 
import fiftyone.types as fot
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from typing import List, Any 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
    )
 
class COCOLoader(Dataset): 
    def __init__(self, coco_datapath: str, coco_labelpath: str):
        super().__init__()
        self.coco_datapath = coco_datapath 
        self.coco_labelpath = coco_labelpath 
        self.full_dataset = fo.Dataset.from_dir(
            dataset_type=fot.COCODetectionDataset, 
            data_path=self.coco_datapath,
            labels_path=self.coco_labelpath 
        ) 
        self.dataset = self.full_dataset.exists("segmentations")
        self.sample_ids = self.dataset.values('id') 

        logging.info(f'COCOLoader | images found: {len(self.dataset)}')

    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample = self.dataset[sample_id] 

        if sample.metadata is None:
            sample.compute_metadata()

        image = np.array(Image.open(sample.filepath).convert('RGB'))
        canvas_width = sample.metadata.width 
        canvas_height = sample.metadata.height 
        labels = [] 
        bounding_boxes = []
        masks = []
        # dataset = {}
        # if sample.segmentations is None: 
        #     continue
        for obj in sample.segmentations.detections: 
            label = obj.label 
            x_norm, y_norm, width_norm, height_norm = obj.bounding_box 
            x_min = int(np.floor(x_norm * canvas_width))
            y_min = int(np.floor(y_norm * canvas_height))
            x_max = int(np.ceil((x_norm + width_norm) * canvas_width))
            y_max = int(np.ceil((y_norm + height_norm) * canvas_height)) 
            bounding_box = [x_min, y_min, x_max, y_max]

            mask = np.int8(obj.mask) if obj.mask is not None else None 
            mh, mw = mask.shape
            overlay_mask = np.int8(np.zeros(shape=(canvas_height, canvas_width)))
            overlay_mask[y_min:y_min+mh, x_min:x_min+mw] = mask

            labels.append(label) 
            bounding_boxes.append(bounding_box)
            masks.append(overlay_mask) 

        return {
            'image_id':sample_id,
            'image': image,
            'canvas_height':canvas_height,
            'canvas_width':canvas_width, 
            'labels':labels, 
            'bounding_boxes':bounding_boxes, 
            'masks':masks
        }
    

class VOCPascalLoader(Dataset): 
    def __init__(
            self,
            split: str = ['train', 'val', 'trainval'],
            images_fp: str = r'C:\Users\ADMIN\Documents\SAM\Research\src\Data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages', 
            annotations_fp: str = r'C:\Users\ADMIN\Documents\SAM\Research\src\Data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations', 
            segmentationsObject_fp: str = r'C:\Users\ADMIN\Documents\SAM\Research\src\Data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationObject', 
            root: str = r'C:\Users\ADMIN\Documents\SAM\Research\src\Data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
            ):
        super().__init__()
        assert split in ("train", "val", "trainval")
        self.split = split
        self.images_fp = images_fp 
        self.annotations_fp = annotations_fp 
        self.segmentationsObject_fp = segmentationsObject_fp 
        self.root = root 

        ids_path = os.path.join(self.root, "ImageSets", "Segmentation", f"{split}.txt")
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"Split file not found: {ids_path}")

        with open(ids_path, "r") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        logging.info(f'VOCPascal Loader | split: {self.split} | images found: {len(self.image_ids)}')

    def __len__(self):
        return len(self.image_ids) 
    
    def __getitem__(self, index):
        image_id = self.image_ids[index] 
        
        image_fp = os.path.join(self.images_fp, image_id + '.jpg')
        image_annotation_fp = os.path.join(self.annotations_fp, image_id + '.xml') 
        image_masks_fp = os.path.join(self.segmentationsObject_fp, image_id + '.png') 

        # existence checks
        if not (os.path.exists(image_fp) and os.path.exists(image_annotation_fp) and os.path.exists(image_masks_fp)):
            logging.error("Missing file for image_id %s: %s %s %s", image_id, image_fp, image_annotation_fp, image_masks_fp)
            raise FileNotFoundError(f"Files missing for {image_id}") 
        
        image = np.array(Image.open(image_fp).convert('RGB')) 
        tree = ET.parse(image_annotation_fp) 
        root = tree.getroot() 
        canvas_width = int(root.find('size').find('width').text) 
        canvas_height = int(root.find('size').find('height').text) 

        labels = []
        bounding_boxes = [] 

        for obj in root.findall('object'): 
            label = obj.find('name').text
            x_min = int(obj.find('bndbox').find('xmin').text)
            y_min = int(obj.find('bndbox').find('ymin').text) 
            x_max = int(obj.find('bndbox').find('xmax').text) 
            y_max = int(obj.find('bndbox').find('ymax').text)  
            bbox = np.array([x_min, y_min, x_max, y_max])
            labels.append(label) 
            bounding_boxes.append(bbox) 


        semantic_image = np.array(Image.open(image_masks_fp)) 
        ids = np.unique(semantic_image)
        ids = ids[(ids != 0) & (ids != 255)]
        masks = [np.array((semantic_image==obj_id)) for obj_id in ids] 
        
        return {
            'image_id':image_id, 
            'image':image, 
            'canvas_height':canvas_height,
            'canvas_width':canvas_width, 
            'labels':labels, 
            'bounding_boxes':bounding_boxes, # (x_min, y_min, x_max_, y_max)
            'masks':masks
        }



class ADE20KLoader(Dataset): 
    def __init__(
            self,
            split: str = 'validation',
            images_fp: str = r'C:\Users\ADMIN\Documents\SAM\Research\src\Data\ADE20K\ADEChallengeData2016\images',
            annotations_fp: str = r'C:\Users\ADMIN\Documents\SAM\Research\src\Data\ADE20K\ADEChallengeData2016\annotations',
            index_fp: str = r'C:\Users\ADMIN\Documents\SAM\Research\src\Data\ADE20K\index_ade20k.pkl'
            ):
        super().__init__()
        assert split in ("training", "validation"), "split must be 'training' or 'validation'"
        self.split = split
        self.images_fp = os.path.join(images_fp, split)
        self.annotations_fp = os.path.join(annotations_fp, split)

        # Load class index: maps integer class id -> human-readable label
        if os.path.exists(index_fp):
            with open(index_fp, 'rb') as f:
                index = pkl.load(f)
            # index['objectnames'] is a list of 150 class name strings (1-indexed)
            self.class_names = ['background'] + list(index['objectnames'])
        else:
            logging.warning("index_ade20k.pkl not found — class labels will fall back to numeric ids.")
            self.class_names = None

        # Collect all image paths recursively (images are nested by scene category)
        self.image_paths = sorted(glob.glob(
            os.path.join(self.images_fp, '**', '*.jpg'), recursive=True
        ))
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found under {self.images_fp}")

        logging.info("ADE20KLoader | split=%s | images found=%d", split, len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def _decode_segmentation(self, seg_path: str):
        """
        Decode an ADE20K *_seg.png into per-instance masks, bounding boxes, and labels.

        ADE20K segmentation encoding (RGB):
            - Class index  : Blue  channel (1-indexed; 0 = background)
            - Instance idx : (Red // 10) * 256 + Green
        
        Returns:
            labels (List[str])         : class name per instance
            bounding_boxes (List[np.ndarray]) : [x_min, y_min, x_max, y_max] per instance
            masks (List[np.ndarray])   : binary boolean mask per instance
        """
        seg = np.array(Image.open(seg_path).convert('RGB'))
        R, G, B = seg[:, :, 0], seg[:, :, 1], seg[:, :, 2]

        class_map    = B.astype(np.int32)                          # H x W, values 0-150
        instance_map = (R.astype(np.int32) // 10) * 256 + G.astype(np.int32)  # H x W

        labels, bounding_boxes, masks = [], [], []

        unique_instances = np.unique(instance_map)
        unique_instances = unique_instances[unique_instances != 0]  # drop background

        for inst_id in unique_instances:
            inst_mask = instance_map == inst_id                     # boolean H x W

            # Derive class from the most common class value within this instance
            class_ids_in_inst = class_map[inst_mask]
            class_id = int(np.bincount(class_ids_in_inst).argmax())

            if class_id == 0:
                continue  # skip background-only instances

            # Resolve label string
            if self.class_names and class_id < len(self.class_names):
                label = self.class_names[class_id]
            else:
                label = str(class_id)

            # Bounding box from mask extents
            rows = np.any(inst_mask, axis=1)
            cols = np.any(inst_mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            labels.append(label)
            bounding_boxes.append(np.array([x_min, y_min, x_max, y_max]))
            masks.append(inst_mask)

        return labels, bounding_boxes, masks

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_id   = os.path.splitext(os.path.basename(image_path))[0]
        
        # Derive annotation path: same relative sub-path, suffix _seg.png
        rel_path   = os.path.relpath(image_path, self.images_fp)
        
        seg_path   = os.path.join(
            self.annotations_fp,
            os.path.dirname(rel_path),
            image_id + '.png'
        )
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Segmentation mask not found: {seg_path}")

        image          = np.array(Image.open(image_path).convert('RGB'))
        canvas_height, canvas_width = image.shape[:2]

        labels, bounding_boxes, masks = self._decode_segmentation(seg_path)

        return {
            'image_id'      : image_id,
            'image'         : image,
            'canvas_height' : canvas_height,
            'canvas_width'  : canvas_width,
            'labels'        : labels,
            'bounding_boxes': bounding_boxes,
            'masks'         : masks
        }