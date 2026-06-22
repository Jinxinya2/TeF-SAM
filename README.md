# TeF-SAM

TeF-SAM is a prototype-guided SAM segmentation model for partially paired image-text medical segmentation. The model uses paired image-text data to build a semantic prototype memory, then performs inference with images only.

This folder is a clean model-only extraction from the original experimental project. It intentionally excludes datasets, logs, visualization scripts, ablation scripts and local training artifacts.

## Directory

```text
opensource_model/
├── README.md
├── environment_requirements.txt
├── requirements-minimal.txt
└── tefsam/
    ├── config.py
    ├── encoders.py
    ├── fusion.py
    ├── layers.py
    ├── model.py
    ├── semantic_memory.py
    └── sam_adapter.py
```

## Model Overview

The model has four main parts:

1. **Visual encoder**: extracts a four-level visual feature pyramid.
2. **External semantic memory**: retrieves semantic prototype responses from image embeddings.
3. **Prompt generator**: produces a coarse mask and high-confidence point prompts.
4. **SAM decoder adapter**: feeds dense mask prompts, point prompts and semantic prompt tokens into the SAM mask decoder.

During inference, text is not required. The semantic memory is provided as an external module and is not included in this open-source model package.

## Core Flow

```text
image
  ├─ visual encoder -> multi-scale visual tokens
  ├─ BiomedCLIP image embedding -> prototype retrieval -> semantic tokens
  └─ frozen SAM image encoder

visual tokens + semantic tokens
  -> coarse mask confidence map
  -> high-confidence point prompts
  -> SAM prompt encoder + SAM mask decoder
  -> segmentation mask
```

## Recommended Prototype Construction

Use `prototype_build_mode="image_key_text_value"`:

1. Encode paired training images as retrieval keys.
2. Encode paired texts as token-level semantic values.
3. Group samples by pseudo class.
4. Initialize class prototypes by farthest-point sampling.
5. Refine prototypes with balanced Sinkhorn assignment.
6. Store averaged text-token memories as prototype values.

This produces an image-queryable semantic memory, which is the key difference from direct text-conditioned segmentation.

## Minimal Usage

```python
import torch
from tefsam import TeFSAM, TeFSAMConfig

cfg = TeFSAMConfig(
    image_size=(224, 224),
    num_classes=6,
    num_prototypes=16,
    prototype_dim=1536,
    clip_dim=768,
    sam_checkpoint="./model/medsam_vit_b.pth",
)

class MySemanticMemory:
    def query(self, image, image_emb=None, image_tokens=None):
        # Return semantic tokens from your private prototype memory.
        return torch.randn(image.shape[0], cfg.num_candidate, cfg.text_dim, device=image.device)

model = TeFSAM(cfg, semantic_memory=MySemanticMemory()).cuda()
batch = {
    "image": torch.randn(2, 1, 224, 224).cuda(),
    # Optional if already precomputed:
    # "image_emb": torch.randn(2, 768).cuda(),
}
out = model(batch)
mask_prob = out["logits"]
```

## Training Notes

Recommended trainable parts:

- multi-scale fusion module,
- coarse mask prompt generator,
- point/mask prompt adapter,
- trainable semantic memory parameters, if your private memory module exposes them.

Frozen parts:

- SAM image encoder,
- SAM prompt encoder,
- base SAM mask decoder weights,
- prototype image-key buffer after construction.

A typical loss is:

```text
loss = DiceCE(segmentation_mask, gt) + lambda * DiceCE(coarse_mask, gt)
```

The original project used `lambda = 0.2`.
