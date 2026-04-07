# **NoisySAM - Evaluate foundation model robustness under perturbations for natural image segmentation [Ongoing]**

- This project evaluates the robustness of foundation segmentation models under different types of noise and image perturbations.

- The goal is to systematically test how segmentation performance changes when images are affected by transformations such as geometric distortions, noise injection, color shifts, and mixed-image augmentations.

---

## **MODELS**

- These models are currently evaluated:
  - SAM
  - MobileSAM
  - FastSAM

## **DATASETS**

- COCO 2017 (validation)
- VOC Pascal 2012 (validation)
- ADE20K (validation)

Please download dataset from this link: [DATASET](https://drive.google.com/drive/folders/13F7XKDBIV-9y4c2LejBK3bt5Upr1HcZs?usp=sharing)

### DATA FORMAT

- All dataloaders in this project return data in a unified format to simplify benchmarking across datasets.
  {
  "image_id": str,
  "image": np.ndarray, # (H, W, 3) RGB image
  "canvas_height": int,
  "canvas_width": int,
  "labels": List[str], # object class names
  "bounding_boxes": List[List[int]], # [x_min, y_min, x_max, y_max]
  "masks": List[np.ndarray] # binary masks (H, W)
  }
- Please read **utils/dataLoader.py** for further understanding

## **CORRUPTIONS & PERTURBATIONS**

To evaluate the robustness of SAM variants, this project implements a rigorous noise injection pipeline via the `NoiseInjection` class. Each corruption can be applied with varying levels of **Severity (1-5)**, simulating real-world environmental challenges and digital artifacts.

<p align="center">
  <img src="./example_image/output.png" width="100%"/>
</p>

### 1. Statistical Noise

These methods simulate sensor limitations and photon counting errors in digital imaging.

- **Gaussian Noise**: Simulates electronic circuit noise by adding random values from a normal distribution.
- **Poisson Noise (Shot Noise)**: Mimicts the statistical nature of light arrival (photons) at the sensor.
- **Salt & Pepper**: Randomly replaces pixels with white or black, simulating bit errors during transmission.
- **Speckle Noise**: Multiplicative noise often seen in medical ultrasound or radar imaging.

---

### 2. Blur Artifacts

Simulates loss of focus, camera movement, or environmental factors that reduce image sharpness.

- **Defocus Blur**: Mimics a camera being out of focus using a disk-shaped kernel.
- **Motion Blur**: Simulates the effect of camera shake or objects moving fast during exposure.
- **Zoom Blur**: Creates a radial blur effect, simulating the lens zooming during a shot.
- **Frosted Glass Blur**: Replicates the scattering of light through a textured translucent surface.

---

### 3. Weather & Environmental Effects

Complex perturbations that simulate challenging outdoor conditions.

- **Fog**: Uses plasma fractals to simulate non-uniform visibility reduction.
- **Snow**: Combines motion-blurred "snowflakes" with a global brightness shift to mimic winter conditions.

---

### 4. Digital & Photographic Distortions

Artifacts introduced during image processing, compression, or lighting changes.

- **Pixelate**: Reduces the spatial resolution of the image, creating a "blocky" effect.
- **JPEG Compression**: Simulates artifacts caused by the lossy JPEG encoding algorithm.
- **Contrast & Brightness**: Evaluates the model's sensitivity to lighting conditions and dynamic range.

---

## Noise Implementation Details

The noise pipeline is highly modular, utilizing the following libraries:

- **Albumentations**: For high-performance statistical noise.
- **OpenCV & SciPy**: For custom kernel-based blurs and transformations.
- **Scikit-Image**: For color space conversions and complex filtering.

You can customize the severity ranges and noise types in `utils/noise_injection.py`.

## MODELS

This project evaluates three major architectures of the Segment Anything family. All models are wrapped in a unified interface inherited from `AbstractLoader` to ensure consistency in the evaluation pipeline.

### Supported Architectures

| Model Family   | Variant         | Checkpoint File        | Backend Library    |
| :------------- | :-------------- | :--------------------- | :----------------- |
| **SAM (v1.0)** | ViT-Base        | `sam_vit_b_01ec64.pth` | `segment_anything` |
| **SAM (v1.0)** | ViT-Huge        | `sam_vit_h_4b8939.pth` | `segment_anything` |
| **MobileSAM**  | Tiny-ViT        | `mobile_sam.pt`        | `mobile_sam`       |
| **FastSAM**    | Small (s)       | `FastSAM-s.pt`         | `ultralytics`      |
| **FastSAM**    | Extra Large (x) | `FastSAM-x.pt`         | `ultralytics`      |

### Model Zoo & Setup

1. **Original SAM (Meta AI)**: High-accuracy foundation models.
   - [Download sam_vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
   - [Download sam_vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

2. **MobileSAM**: Optimized for mobile and CPU efficiency, ~40MB in size.
   - [Download mobile_sam.pt](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt)

3. **FastSAM**: Based on YOLOv8 instance segmentation, optimized for real-time inference.
   - [Download FastSAM-s/x](https://github.com/CASIA-IVA-Lab/FastSAM)

---

### Model Loading Logic

The `utils/modelLoader.py` provides a standardized way to interact with different model backends. Each class implements `_set_image()` and `_predict()` methods to abstract away the specific API calls of each library.

### Example: Initializing a Model

You can initialize any model variant and move it to the appropriate device (CUDA is automatically detected) as follows:

```python
from utils.modelLoader import SAM1, MobileSAM, FastSAMModel

# Initialize SAM ViT-B
sam_b = SAM1(
    model_name="SAM_b",
    model_type="vit_b",
    checkpoint="path/to/sam_vit_b.pth"
)

# Initialize FastSAM
fast_sam = FastSAMModel(
    model_name="FastSAM_x",
    model="path/to/FastSAM-x.pt"
)
```

## USAGE

The evaluation pipeline is organized around the Experiment class in run.py. It takes four main inputs: a model wrapper, a dataset loader, a dictionary of corruption functions, and a list of severity levels. During evaluation, the pipeline iterates over every (noise_type, severity) pair, corrupts each image, runs segmentation with the selected model, computes metrics, and saves both masks and tabular results.

1. Prepare the dataset

Each dataset loader returns samples in a unified format with the following keys:

```python
{
    "image_id": str,
    "image_path": str,
    "image": np.ndarray,          # RGB image, shape (H, W, 3)
    "canvas_height": int,
    "canvas_width": int,
    "labels": List[str],
    "bounding_boxes": List[List[int]],  # [x_min, y_min, x_max, y_max]
    "masks": List[np.ndarray]           # binary masks, shape (H, W)
}

```

The project includes loaders for COCO, VOC Pascal, and ADE20K, and each loader extracts bounding boxes and masks in a consistent format so they can be passed directly into the same evaluation loop.

2. Initialize a model

All supported models follow a shared interface with \_set_image() and \_predict() methods. The repository currently supports SAM, MobileSAM, and FastSAM through the wrappers defined in modelLoader.py.

Example:

```python

from utils.modelLoader import SAM1, MobileSAM, FastSAMModel

sam_b = SAM1(
    model_name="SAM_b",
    model_type="vit_b",
    checkpoint="path/to/sam_vit_b_01ec64.pth"
)

mobile_sam = MobileSAM(
    model_name="MobileSAM",
    model_type="vit_t",
    checkpoint="path/to/mobile_sam.pt"
)

fast_sam = FastSAMModel(
    model_name="FastSAM_x",
    model="path/to/FastSAM-x.pt"
)
```

The SAM and MobileSAM wrappers call the corresponding predictors with a bounding box prompt in [x1, y1, x2, y2] format, while FastSAM uses its own prompt-based predictor internally.

3. Define corruptions

Corruptions are implemented in noise_injection.py and are applied per image before inference. The severity level is an integer from 1 to 5, and each corruption has its own severity-dependent parameter schedule. The current pipeline includes Gaussian noise, Poisson noise, salt-and-pepper noise, speckle noise, blur variants, fog, snow, brightness, contrast, pixelation, JPEG compression, and other perturbations implemented in the noise module.

Example:

```python
from utils.noise_injection import NoiseInjection

noise = NoiseInjection()

noise_dict = {
    "gaussian": noise._inject_gaussian_noise,
    "poisson": noise._inject_poisson_noise,
    "salt_pepper": noise._inject_salt_and_pepper_noise,
    "speckle": noise._inject_speckle_noise,
    "motion_blur": noise._motion_blur,
    "fog": noise.fog,
    "snow": noise._inject_snow,
    "jpeg": noise._inject_JPEG,
}

```

4. Run the experiment

The experiment is executed by creating an Experiment object and calling \_evaluate(). The evaluation loop processes every image, applies corruption, predicts masks from bounding boxes, computes metrics, and saves results to disk.

Example:

```python
from run import Experiment

configs = {
    "experiment_tag": "coco_fastsam_noise_eval",
    "dataset_name": "COCO",
    "model_name": "FastSAM_x",
    "output_dir": "./results"
}

experiment = Experiment(
    model=fast_sam,
    dataset=coco_dataset,
    noise_dict=noise_dict,
    severities=[1, 2, 3, 4, 5],
    configs=configs
)

experiment._evaluate()

```

---

## OUTPUT

The experiment writes three kinds of outputs to output_dir:

1. Metadata JSON

A metadata file is created at:

```python
{output_dir}/{experiment_tag}_metadata.json
```

This file stores the experiment timestamp, the configuration dictionary, the selected severities, the corruption names, the metric names, and a completion status flag.

2. Per-instance metrics CSV

A CSV file is created at:

```python
{output_dir}/{experiment_tag}_metrics.csv
```

Each row corresponds to one object instance under one corruption setting. The saved columns are:

```
image_id | ann_id | model_name | noise_type | severity | label | x_min | y_min | x_max | y_max | iou | dice | hd95 | precision | recall | confidence | is_bad_case | is_failure_case | failure_type |
| pred_mask_path | image_path

```

These fields are written directly by \_add_to_results() in run.py.

3. Predicted masks

Predicted masks are saved as PNG files under:

```
{output_dir}/masks/{dataset_name}/{model_name}/{noise_type}/sev{severity}/{image_id}/{ann_id}_{label}_pred_mask.png
```

If a prediction fails, no mask is written and the CSV row is marked as a failure case with pred_mask_path = "NONE".

---

METRICS

The evaluation pipeline computes the following metrics for every predicted mask against its ground-truth mask:

- IoU: Intersection over Union
- Dice: Dice similarity coefficient
- HD95: 95th percentile Hausdorff distance
- Precision
- Recall
- Failure handling

A prediction is treated as a failure if the predicted mask is None or if it is an empty mask. In those cases, the metrics are set to zero and the row is marked with a failure reason such as None_Mask or Empty_Mask.

Bad-case flag

A prediction is flagged as a bad case when Dice < 0.3. This threshold is used only for analysis and does not stop the evaluation process.

Confidence score

The confidence field stores the first score returned by the model wrapper when available. For SAM and MobileSAM, this comes from the predictor output score array, and for FastSAM it is taken from the box confidence scores.
