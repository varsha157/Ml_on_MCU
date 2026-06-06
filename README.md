# Traffic Image Segmentation on STM32M4 & MAX78000

Semantic segmentation of urban traffic scenes deployed on two microcontroller platforms — the **STM32M4** and **MAX78000** — as part of the *Machine Learning on Microcontrollers* course at ETH Zurich (January 2024).

**Authors:** Phani Jayanth Jonnalagadda, Varsha Jayaprakash

---

## Overview

This project explores the full pipeline of training, optimizing, and deploying lightweight semantic segmentation models on resource-constrained MCUs. Models are trained on the [Cityscapes dataset](https://www.cityscapes-dataset.com/) and adapted for embedded inference by applying quantization-aware training (QAT), pruning, and — for the MAX78000 — a data folding technique to handle higher-resolution inputs.

The two target platforms differ substantially in their compute profiles:

- **STM32M4** (ARM Cortex-M4 @ 80 MHz) — runs float32/pruned models, higher inference latency but flexible.
- **MAX78000** (Analog Devices, CNN hardware accelerator) — runs 8-bit quantized models, extremely fast inference via its dedicated CNN engine (85–133 MACs/cycle throughput).

---

## Dataset: Cityscapes

| Property | Details |
|---|---|
| Focus | Semantic understanding of urban street scenes |
| Classes | 30 original classes, remapped to 3 or 4 for MCU deployment |
| Cities | 50 cities |
| Annotated images | 5,000 |
| Resolution | 1024 × 2048 PNG, resized to 80×80 (STM32) or 352×352 → folded (MAX78000) |

**Known limitations:** Cityscapes images are captured only during daylight and in good weather conditions, which limits model generalization to nighttime, rain, snow, and fog scenarios.

### Class Mappings

| Setup | Classes |
|---|---|
| 3-class | Void, Vegetation, Vehicle |
| 4-class (STM32M4) | Void, Vegetation, Vehicle, Human |
| 4-class (MAX78000) | Void, Vegetation, Vehicle, Building |

---

## Models

Four architectures were implemented and benchmarked across the two platforms.

### 1. Simple Encoder-Decoder (`simple`) — STM32M4
- Input: 3-channel RGB, 40×40
- 5× Conv2D (3×3) + 1× Conv2D (1×1)
- 2× MaxPool (2×2) + 2× ConvTranspose2D upsampling
- ~17K parameters

### 2. Enhanced Encoder-Decoder (`ourmodelcomplex`) — STM32M4 & MAX78000
- Input: 3-channel RGB, 80×80
- 7× Conv2D (3×3) + 1× Conv2D (1×1), all with BatchNorm + ReLU
- 3× MaxPool (2×2) + 3× ConvTranspose2D upsampling
- ~52K parameters
- Implemented using Analog Devices `ai8x` layers for MAX78000 compatibility

### 3. Small U-Net (Analog Devices Inc.) — STM32M4 & MAX78000
- Input: 3-channel RGB, 80×80
- U-Net architecture with 3 skip connections and concatenations
- ~62K parameters

### 4. Large U-Net (Analog Devices Inc.) — MAX78000
- Input: 3-channel RGB, 352×352, folded by factor 4 → 48 channels, 88×88
- 8× Conv2D (3×3) + 7× Conv2D (1×1)
- 3 skip connections + 3 upsampling layers
- Output unfolded back to 4×352×352
- ~277K parameters

---

## Data Augmentation

The following augmentations are applied during training (implemented in `cityscapes.py`):

- `RandomHorizontalFlip` (p=0.4)
- `ColorJitter` (brightness, contrast, saturation)
- `RandomGrayscale` (p=0.2)
- `GaussianBlur` (kernel=3)
- `RandomInvert` (p=0.1)
- `RandomAutocontrast` (p=0.1)
- `RandomSolarize` (p=0.1)

For MAX78000 (folded pipeline), `ai8x.fold(fold_ratio=4)` is applied after normalization.

---

## Training & Optimization

### STM32M4
- Training with float32 precision
- Unstructured pruning with dynamic sparsity
- Quantization-Aware Training (QAT) applied post-pruning

### MAX78000
- QAT with **8-bit weights** (`qat_policy_cityscapes.yaml`)
  - QAT starts at epoch 15
- Learning rate schedule (`schedule-cityscapes.yaml`):
  - `MultiStepLR` with milestones at epochs 10 and 30, γ=0.5
  - Training runs for 40 epochs
- Hardware mapping via `ourmodelcomplex.yaml` (processor and memory offset assignments for the CNN accelerator)

---

## Results

### Pixel-to-Pixel Accuracy and mIoU

#### STM32M4 (Unstructured Pruning + QAT float32)

| Model | Classes | Pixel Accuracy (%) | mIoU (%) |
|---|---|---|---|
| Simple | 3 | 79.74 | 77.85 |
| Simple | 4 | 79.14 | 78.59 |
| Enhanced | 3 | 81.12 | 79.20 |
| Enhanced | 4 | 80.22 | 79.63 |
| Small U-Net | 3 | 82.03 | 80.08 |
| Small U-Net | 4 | 80.82 | 80.84 |

#### MAX78000 (QAT, 8-bit weights)

| Model | Classes | Pixel Accuracy (%) | mIoU (%) |
|---|---|---|---|
| Enhanced | 3 | 84.408 | 89.667 |
| Enhanced | 4 | 69.715 | 81.95 |
| Small U-Net | 3 | 84.635 | 89.962 |
| Small U-Net | 4 | 71.075 | 85.271 |
| Large U-Net | 3 | 86.43 | 91.151 |
| Large U-Net | 4 | 75.654 | 87.833 |

> Reference: Analog Devices Inc. reports 91.05% pixel accuracy and 84.24% mIoU on CamVid (4 classes) with 95.1 ms inference time.

### Hardware Benchmarks (3-Class)

| Metric | Simple (STM32) | Enhanced (STM32) | Enhanced (MAX78000) | Small U-Net (STM32) | Small U-Net (MAX78000) | Large U-Net (MAX78000) |
|---|---|---|---|---|---|---|
| Parameters | 17,323 | 52,259 | 52,104 | 62,487 | 62,940 | 277,152 |
| MACs (M) | 2.167 | 5.712 | 22.426 | 4.56 | 34.125 | 619.769 |
| RAM | 66.03 KB | 69.06 KB | — | 55.26 KB | — | — |
| Flash | 58.67 KB | 97.92 KB | — | 109.89 KB | — | — |
| Quantized Memory | — | — | 51.08 KB | — | 62.67 KB | 274.59 KB |
| Cycles (M) | 15.68 | 17.36 | 0.261 | 10.48 | 0.365 | 4.638 |
| Inference Time (ms) | 196 | 217 | 5.216 | 131 | 7.3 | 92.753 |
| Throughput (MACs/Cycle) | 0.138 | 0.329 | 85.989 | 0.432 | 93.486 | 133.638 |

---

## Repository Structure

```
Ml_on_MCU/
├── cityscapes.py                        # Dataset loader, label remapping, augmentation pipelines
├── ourmodelcomplex.py                   # Enhanced encoder-decoder model (ai8x-compatible)
├── ourmodelcomplex.yaml                 # MAX78000 hardware layer mapping (processor/memory offsets)
├── qat_policy_cityscapes.yaml           # QAT config (starts epoch 15, 8-bit weights)
├── schedule-cityscapes.yaml             # LR schedule (MultiStepLR, milestones 10 & 30, 40 epochs)
├── cityscapes3_complexunet_s80-q_pth.tar  # Trained checkpoint — 3-class Enhanced model
├── cityscapes4_complexunet_s80-q_pth.tar  # Trained checkpoint — 4-class Enhanced model
└── README.md
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision pillow numpy matplotlib
```

This project also depends on the [ai8x-training](https://github.com/analogdevicesinc/ai8x-training) framework from Analog Devices Inc. for MAX78000-compatible layers (`ai8x.FusedConv2dBNReLU`, `ai8x.ConvTranspose2d`, etc.) and for QAT/synthesis support.

### Dataset

Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/) and place it at:

```
<data_dir>/cityscapes/leftImg8bit/{train,val}/
<data_dir>/cityscapes/gtFine/{train,val}/
```

### Training (STM32M4 / Standard Pipeline)

```bash
python train.py \
  --model ourmodelcomplex \
  --dataset cityscapes \
  --schedule schedule-cityscapes.yaml \
  --qat-policy qat_policy_cityscapes.yaml \
  --data <data_dir>
```

### Training (MAX78000 / Folded Pipeline)

Use the `cityscapes_folded` dataset entry (48-channel folded input at 88×88) with the same training script and the `ourmodelcomplex.yaml` hardware mapping for synthesis.

### Loading a Checkpoint

```python
import torch
from ourmodelcomplex import ourmodelcomplex

model = ourmodelcomplex(num_classes=4)
checkpoint = torch.load('cityscapes4_complexunet_s80-q_pth.tar', map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
```

---

## Limitations & Future Work

- The number of supported segmentation classes is limited by MCU memory constraints.
- Cityscapes is daylight-only and good-weather-only; models degrade on night scenes, rain, snow, and fog.
- float32 inference on STM32M4 gives better accuracy but at the cost of high memory usage.
- Inference times are not yet suitable for real-time autonomous driving (target: <33 ms).
- Future directions: transfer learning from broader datasets, panoptic segmentation, night/adverse-weather domain adaptation.

---

## References

- [L³U-Net: Low-Latency Lightweight U-Net for Parallel CNN Processors](https://arxiv.org/pdf/2203.16528) — Analog Devices Inc., 2022
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [ai8x-training framework](https://github.com/analogdevicesinc/ai8x-training) — Analog Devices Inc.
- ETH Zurich, Machine Learning on Microcontrollers course, January 2024
