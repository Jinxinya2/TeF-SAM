# TeF-SAM Model Structure

This directory contains only the model-side code extracted from the experimental STL-SAM project.

## Forward Path

1. **Image feature extraction**
   - `VisionPyramidEncoder` loads a ConvNeXt-style visual backbone.
   - Four feature maps are returned as a visual pyramid.

2. **Multi-scale prompt feature fusion**
   - `MultiScaleFeatureFusion` projects the four visual maps to `sam_prompt_dim`.
   - Top-down, bottom-up and adaptive scale fusion produce `visual_tokens`.

3. **Prototype semantic retrieval**
   - `SemanticMemory` is an interface for an external image-queryable semantic memory.
   - The retrieved values are semantic tokens derived from your private prototype construction pipeline.
   - During inference, no text input is required.

4. **Coarse mask and point prompt generation**
   - `MaskPromptGenerator` cross-attends `visual_tokens` to retrieved semantic tokens.
   - It predicts a coarse mask confidence map.
   - Positive points are selected from high-confidence local maxima, then diversified by farthest-point sampling.

5. **SAM mask decoder**
   - `SAMPromptDecoder` sends three prompt types into SAM:
     - coarse dense mask prompt,
     - selected point prompts,
     - a semantic sparse token projected from prototype response.
   - The SAM mask decoder is LoRA-tuned; the SAM prompt encoder and image encoder are frozen.
   - `decoder_image_source` controls the image feature supplied to the mask decoder:
     - `sam_encoder`: frozen SAM image encoder output.
     - `dense_prompt`: fused visual prompt feature resized to SAM embedding size.

## Prototype Construction

Recommended mode: `image_key_text_value`.

1. Encode paired training images with BiomedCLIP image encoder.
2. Group samples by pseudo class label.
3. Initialize each class prototype with farthest-point sampling in image embedding space.
4. Refine prototypes with balanced Sinkhorn assignment.
5. Store paired text-token averages as prototype semantic memory.
6. Optionally initialize the trainable semantic prototype part from text-token means.

The result is an image-queryable semantic memory. The construction implementation is intentionally not included in this open-source model package.
