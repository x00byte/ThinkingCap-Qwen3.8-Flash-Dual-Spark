# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for x00byte's ThinkingCap deployment; see THIRD_PARTY.md.
"""Conservative projection fusion admission for quantized QSA checkpoints."""


def can_fuse_qsa_indexer_projection(vllm_config):
    # A single merged linear cannot preserve different quantization schemes.
    # Keep native fusion for unquantized, non-LoRA models only; quantized
    # checkpoints retain their original QKV and indexer weights and kernels.
    return vllm_config.lora_config is None and vllm_config.quant_config is None


def configure_qsa_projection_mapping(model, vllm_config):
    if can_fuse_qsa_indexer_projection(vllm_config):
        return
    separate = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    model.packed_modules_mapping = model.packed_modules_mapping | separate
    quant_config = vllm_config.quant_config
    if quant_config is not None:
        # configure_quant_config ran before model construction. Give this
        # instance its own mapping; never mutate a model class dictionary.
        quant_config.packed_modules_mapping = (
            quant_config.packed_modules_mapping | separate
        )
