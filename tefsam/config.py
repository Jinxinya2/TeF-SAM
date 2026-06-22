from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TeFSAMConfig:
    image_size: Tuple[int, int] = (224, 224)
    device: str = "cuda"

    vision_type: str = "facebook/convnext-tiny-224"
    biomedclip_path: str = "./model"
    biomedclip_tokenizer: str = "chuhac/BiomedCLIP-vit-bert-hf"

    num_classes: int = 6
    num_prototypes: int = 16
    num_candidate: int = 3
    prototype_dim: int = 1536
    clip_dim: int = 768
    text_dim: int = 768
    text_len: int = 256
    text_feature_dim: int = 768
    agg: str = "attention"

    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_b"
    sam_prompt_dim: int = 256
    sam_image_embedding: int = 64
    decoder_image_source: str = "sam_encoder"
    sam_encoder_batch_size: int = 1

    point_topk: int = 8
    point_threshold: float = 0.9
    point_quantile: float = 0.995
    use_local_max: bool = True
    use_quantile_threshold: bool = True
    max_candidates: int = 1024
    initial_point_topk: int = 4
    correction_iters: int = 3
    correction_num_points: int = 2

    lora_rank: int = 8
    lora_alpha: float = 1.0
    mask_prompt_loss_weight: float = 0.2
