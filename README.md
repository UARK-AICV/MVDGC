# MVCyDe: Multi-View Cylindrical Detection

**3D Bird's-Eye-View & 2D Image-View Person Detection from Multiple Cameras**

<p align="center">
  <img src="abstract.png" width="900" alt="MVCyDe Framework Overview"/>
</p>

> **MVCyDe** detects people jointly in bird's-eye-view (BEV) ground space and per-camera image space using a set of learnable 3D cylindrical queries. Each query is initialized across the BEV plane, projected into every camera view to sample multi-view features via sparse deformable attention, and iteratively refined to produce accurate ground-plane positions and per-view bounding boxes — all in a single forward pass, with no depth estimation or explicit 3D reconstruction.

---

## How It Works

| Step | Description |
|------|-------------|
| **1 — Cylindrical Query Initialization** | A grid of 3D cylindrical queries is placed across the BEV. Each cylinder encodes a person hypothesis with a ground location (x, y), radius, and height. |
| **2 — Feature Sampling & Aggregation** | Each query is projected into all camera views. Multi-view image features are sampled at the projected locations via multi-scale deformable attention and fused across views. |
| **3 — Cylinder Position & Shape Refinement** | Fused features refine each cylinder's ground position and shape through multiple decoder layers, producing aligned BEV detections and 2D image-view bounding boxes simultaneously. |

---

## Supported Datasets

| Dataset | Cameras | Frames | Scene | Config |
|---------|---------|--------|-------|--------|
| [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) | 7 | 400 | Outdoor plaza | `configs/wildtrack.yaml` |
| [MultiviewX](https://github.com/hou-yz/MultiviewX) | 6 | 400 | Synthetic | `configs/multiviewx.yaml` |
| [GMVD](https://github.com/kev-in-ta/GMVD) | 8 | 2000 | Outdoor campus | `configs/gmvd.yaml` |

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd MVDGC
```

### 2. Create a Python 3.8 environment

```bash
conda create -n mvcyde python=3.8 -y
conda activate mvcyde
```

### 3. Install PyTorch 2.1 with CUDA 12.1

```bash
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install mmcv-full (required for the ViT backbone)

```bash
pip install mmcv-full==1.7.2 -f \
    https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install mmengine==0.10.7
```

> **Note:** `mmcv-full` must match your CUDA and PyTorch versions exactly. If the prebuilt wheel is not available, build from source: `MMCV_WITH_OPS=1 pip install mmcv-full==1.7.2`

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 6. Build the custom CUDA operator

The multi-scale deformable attention CUDA kernel must be compiled before training.

```bash
cd models/ops
bash make.sh
cd ../..
```

Verify the build succeeded:

```bash
python models/ops/test.py
```

### 7. Prepare datasets

Set the `ROOT` path in the config file for your chosen dataset:

```yaml
# configs/wildtrack.yaml
DATASET:
  ROOT: '/path/to/Wildtrack_dataset'

# configs/multiviewx.yaml
DATASET:
  ROOT: '/path/to/MultiviewX'

# configs/gmvd.yaml
DATASET:
  ROOT: '/path/to/GMVD'
```

---

## Training

```bash
# Wildtrack
python main_sparse_deform_batch.py \
    --cfg ./configs/wildtrack.yaml \
    --exp_name wildtrack_run1

# MultiviewX
python main_sparse_deform_batch.py \
    --cfg ./configs/multiviewx.yaml \
    --exp_name multiviewx_run1

# GMVD
python main_sparse_deform_batch.py \
    --cfg ./configs/gmvd.yaml \
    --exp_name gmvd_run1
```

Checkpoints are saved to `output/` with the naming pattern `<dataset>_epoch_<N>.pth`.

### Config overrides via command line

Any config key can be overridden without editing the YAML:

```bash
python main_sparse_deform_batch.py \
    --cfg ./configs/gmvd.yaml \
    --exp_name 1 \
    TRAIN.LR=0.0002 \
    DECODER.num_instance=864
```

---

## Evaluation

Set the checkpoint name inside `test_sparse_deform.py`:

```python
# test_sparse_deform.py  (line ~80)
weight_name = "wildtrack_epoch_28.pth"   # file must be inside output/
```

Then run:

```bash
python test_sparse_deform.py --cfg ./configs/wildtrack.yaml
```

### Metrics

Results are reported using **CLEAR MOD** metrics:

| Metric | Description |
|--------|-------------|
| **Recall** | Percentage of ground-truth persons correctly detected |
| **Precision** | Percentage of predictions that match a ground-truth person |
| **MODA** | Multiple Object Detection Accuracy — penalises both missed and false detections |
| **MODP** | Multiple Object Detection Precision — measures localisation quality of matched detections |

---

## Project Structure

```
MVDGC/
├── configs/
│   ├── config.py               # Master config with all defaults
│   ├── wildtrack.yaml
│   ├── multiviewx.yaml
│   └── gmvd.yaml
│
├── models/
│   ├── bev_sparse_batch.py         # BEVGEO model + SetCriterion loss
│   ├── bev_sparse_batch_decoder.py # BEVDecoder + deformable encoder
│   ├── encoder.py                  # CNN backbone (ResNet50 FPN)
│   ├── VIT_encoder.py              # ViT backbone via mmdet (CoDETR)
│   ├── matcher.py                  # Hungarian set-prediction matcher
│   ├── position_encoding.py        # Sinusoidal positional encoding
│   └── ops/                        # Custom CUDA deformable attention
│       ├── make.sh
│       ├── functions/
│       ├── modules/
│       └── src/
│
├── dataset/
│   ├── wildtrack.py
│   ├── multiviewx.py
│   └── gmvd.py
│
├── utils/
│   ├── metrics.py          # CLEAR MOD, NMS, mAP functions
│   ├── utils.py            # Distributed training helpers
│   ├── geom.py             # Camera geometry
│   ├── vox.py              # Voxel utilities
│   └── basic.py            # Tensor utilities
│
├── main_sparse_deform_batch.py   # Training entry point
├── test_sparse_deform.py         # Evaluation entry point
├── requirements.txt
└── README.md
```

---

## Environment Reference

Tested with the following configuration:

| Component | Version |
|-----------|---------|
| Python | 3.8 |
| PyTorch | 2.1.0+cu121 |
| torchvision | 0.16.0+cu121 |
| CUDA | 12.1 |
| mmcv-full | 1.7.2 |
| mmengine | 0.10.7 |
| numpy | 1.24.4 |
| scipy | 1.10.1 |
| timm | 1.0.15 |
| opencv-python | 4.11.0.86 |
| Pillow | 10.4.0 |
| easydict | 1.13 |
| pycocotools | 2.0.7 |
| scikit-learn | 1.3.2 |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{mvdgc2026,
  title   = {MVDGC: Joint 3D and 2D Multi-view Pedestrian Detection via Dual Geometric Constraints},
  author  = {},
  year    = {2024}
}
```

---

## Acknowledgements

This project builds upon:
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) — sparse deformable attention
- [mmdetection](https://github.com/open-mmlab/mmdetection) — ViT backbone and SFP neck
- [MVDeTr](https://github.com/hou-yz/MVDeTr) — multi-view detection framework

Please refer to their respective licenses when using this code.
