# Comparison: `dinov3_decoder_ViT.py` vs. `dinov3_decoder_ViT_old.py`

This document summarizes the architectural and implementation differences between the old and new versions of the DINOv3 decoder.

## 1. Transformer Decoder Blocks & Conditioning Mechanism
**Old Version:**
* Used a single `DecoderBlock` implementation.
* The lighting condition (`light_emb`) was injected using a simple constraint scaling mechanism: the input tokens were scaled by `(1 + param)` where `param = tanh(affine(light_emb)) * alpha`.

**New Version:**
* Introduces two distinct decoder blocks to support different conditioning strategies, controlled by a new `conditioning` argument:
  1. **`DecoderBlock_AdaLN`**: Injects lighting conditions using Adaptive Layer Normalization (AdaLN-Zero). It predicts scale and shift parameters for LayerNorms, and gating parameters for residual connections.
  2. **`DecoderBlock_CrossAttn`**: Injects lighting conditions using a newly added `CrossAttention` module (with an added `affine_scale` parameter).

## 2. Positional Encodings
**Old Version:**
* Used a custom `PositionalEncoding2D` class (2D Sinusoidal Encodings).

**New Version:**
* Removed `PositionalEncoding2D` and replaced it with **Rotary Position Embeddings (RoPE)** via the `RopePositionEmbedding` module from the DINOv3 layers.

## 3. Convolutional Components & Upsampling
**Old Version:**
* Hand-rolled `ConvModule` and `ResidualBlock` classes within the file.
* Used a monolithic `ConvRefinementHead` that sequentially upsampled the output of the final transformer block and refined it with residual blocks.

**New Version:**
* Offloaded basic Convolutional layers to a separate file (`.conv_modules`), importing `ConvLayer`, `ConvPixelShuffleUpSampleLayer`, and `InterpolateConvUpSampleLayer`.
* Built a new **`FeaturePyramidNeck`** that fuses multi-scale features natively instead of purely sequential upsampling. It upsamples all features to half the target resolution and concatenates them.
* Replaced the heavy refinement head with a lighter **`final_head`** consisting of a simple 2-layer CNN after the pyramid neck.

## 4. Input & Feature Aggregation
**Old Version:**
* Only passed the deepest (single) latent output to the transformer stack, applying the normalized hypercolumn fusion method.

**New Version:**
* The transformer pass sequentially processes blocks, but now explicitly iterates and fuses features from different skip connections (multiple layers from the encoder) at each block level using `fusion_proj`.
* Decoder layer structure requires skip connections to be matched iteratively up the pyramid, passing the stacked results into the feature pyramid neck.

## 5. Cleanup
**Old Version:**
* Included factory functions for `dinov3_decoder_small` and `dinov3_decoder_large`.
**New Version:**
* Removed the `small` and `large` specific setups, keeping only the highly parameterized base variant (`dinov3_decoder_base`).
