# TeF-SAM

This repository contains the open-source code for TeF-SAM.

## Installation

Python 3.8+ is supported. Install a PyTorch build appropriate for the local CUDA
version first, then install the vendored SAM package and TeF-SAM:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ./third_party/segment_anything
pip install -e .
```

## Required weights

The frozen PRISM artifact is included at
`artifacts/qata_cov19_prism_3x2_r15.safetensors`. Two additional weight sources
are required:

1. BiomedCLIP base checkpoint `chuhac/BiomedCLIP-vit-bert-hf`. The artifact
   contains only the adapted vision delta, not the base model.
2. MedSAM ViT-B checkpoint at `weights/medsam_vit_b.pth`. Obtain
   `medsam_vit_b.pth` from the official
   [MedSAM repository](https://github.com/bowang-lab/MedSAM).

The ConvNeXt backbone is pinned to Hugging Face revision
`6166b7613034066690a621d8bf25ffdf181a34f0`; the runtime passes this revision
explicitly when loading `facebook/convnext-tiny-224`.

An optional artifact mirror is available on
[Aliyun Drive](https://www.alipan.com/s/5sdQZ5z3DSJ).

For an offline local BiomedCLIP copy, override the checkpoint path:

```bash
--set biomedclip_checkpoint=/path/to/BiomedCLIP \
--set biomedclip_local_files_only=true
```

## Data layout

Prepare QaTa-COV19 as follows. Reports are not required by the released
segmentation runtime. The exact experiment partition is versioned at
`splits/qata_cov19_split.json`; the public configuration references this file
directly, so it must not be replaced by a newly generated split.

```text
splits/
└── qata_cov19_split.json
data/QaTa/
├── images/
│   └── <sample_id>.png
└── masks/
    └── <sample_id>.png
```

`splits/qata_cov19_split.json` contains distinct `train`, `val`, and `test`
arrays. The training command rejects a configuration that uses the test split
for checkpoint selection.

## Train

```bash
python scripts/train.py --config configs/qata_cov19_r15.yaml
```

## Evaluate

```bash
python scripts/evaluate.py \
  --config configs/qata_cov19_r15.yaml \
  --checkpoint checkpoints/qata_cov19/tefsam_r15_prism_3x2.ckpt
```

## Public project structure

```text
TeF-SAM-public/
├── tefsam/
│   ├── models/
│   │   ├── frozen_prism.py
│   │   ├── segmenter.py
│   │   ├── multiscale_fusion.py
│   │   ├── sam_adapter.py
│   │   └── lora.py
│   ├── engine/
│   └── data/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── download_artifact.py
├── configs/
│   └── qata_cov19_r15.yaml
├── splits/
│   └── qata_cov19_split.json
└── artifact_manifests/
    └── qata_cov19_prism_3x2_r15.json
```
