"""Opt-in Qwen4Exp DFlash2 bridge and private draft quantization.

Target features follow the model author's completed-layer/HC-mean contract.
Pending HC residual additions are materialized only inside the feature output;
the target's delayed execution and weights remain untouched.
"""
import json
import os
import types
from pathlib import Path

import torch
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

log = init_logger('vllm.thinkingcap_dflash2')
TAPS = (6, 16, 26, 36, 46)


def validate_aux_layers(config, layers):
    if os.environ.get('THINKINGCAP_ENABLE_DFLASH2') != '1':
        raise ValueError('ThinkingCap DFlash2 feature capture requires explicit opt-in')
    if (config.hidden_size, config.hc_count, config.num_hidden_layers) != (2560, 4, 48):
        raise ValueError('Unexpected Flash Next HC geometry')
    if tuple(layers) != TAPS:
        raise ValueError(f'DFlash2 requires completed layers {[x - 1 for x in TAPS]}')
    return TAPS


def validate_pair(cfg):
    if os.environ.get('THINKINGCAP_ENABLE_DFLASH2') != '1':
        raise NotImplementedError('Qwen4Exp DFlash2 is experimental; explicit opt-in required')
    spec = cfg.speculative_config
    draft = spec.draft_model_config.hf_config
    target = cfg.model_config.hf_text_config
    validate_aux_layers(target, TAPS)
    if draft.architectures != ['DFlash2DraftModel']:
        raise ValueError('Only the Flash Next DFlash2 architecture is qualified for this bridge')
    if (draft.hidden_size, draft.vocab_size, draft.num_target_layers) != (2560, target.vocab_size, 48):
        raise ValueError('Draft and target geometry mismatch')
    if tuple(draft.dflash_config['target_layer_ids']) != tuple(x - 1 for x in TAPS):
        raise ValueError('Draft target feature taps mismatch')
    if spec.num_speculative_tokens != 3:
        raise ValueError('Initial Flash Next DFlash2 bridge requires exactly three drafts')
    pc = cfg.parallel_config
    if pc.pipeline_parallel_size != 1 or pc.prefill_context_parallel_size != 1 or pc.decode_context_parallel_size != 1:
        raise ValueError('Initial bridge supports TP with PP=CP=1')
    if cfg.model_config.max_model_len > draft.max_position_embeddings:
        if os.environ.get('THINKINGCAP_DFLASH_EXTEND_CONTEXT') != '1':
            raise ValueError('Draft positional cache is shorter than the requested target context; do not silently reduce the API context')
        rope = getattr(draft, 'rope_parameters', {}) or {}
        if rope.get('rope_type') != 'default' or rope.get('rope_theta') != 10000 or draft.sliding_window != 2048:
            raise ValueError('Unexpected draft positional contract')
        if cfg.model_config.max_model_len > 409600:
            raise ValueError('Context extension is capped at the existing production contract')
        original_limit = draft.max_position_embeddings
        # Extend only the position lookup table. The draft's RoPE frequencies,
        # weights, local window and the target's existing YaRN settings stay
        # unchanged. Acceptance beyond the trained limit must be measured.
        draft.max_position_embeddings = cfg.model_config.max_model_len
        spec.draft_model_config.max_model_len = cfg.model_config.max_model_len
        log.info('DFlash2 positional cache extended from %d to %d; original RoPE theta=10000',
                 original_limit, draft.max_position_embeddings)
    if cfg.scheduler_config.async_scheduling:
        raise ValueError('Initial DFlash2 qualification uses synchronous scheduling')


@triton.jit
def _completed_feature(residual, block, injection, out,
                       RSTR: tl.constexpr, BSTR: tl.constexpr, ISTR: tl.constexpr,
                       H: tl.constexpr, HC: tl.constexpr, TILE: tl.constexpr,
                       HAS_BLOCK: tl.constexpr, HAS_INJ: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    stream = tl.arange(0, HC)
    x = tl.load(residual + row * RSTR + stream[:, None] * H + col[None, :],
                mask=col[None, :] < H, other=0).to(tl.float32)
    if HAS_BLOCK:
        b = tl.load(block + row * BSTR + col, mask=col < H, other=0).to(tl.float32)
        if HAS_INJ:
            g = tl.load(injection + row * ISTR + stream).to(tl.float32)
            g = 2.0 * tl.sigmoid(g / HC)
            x += g[:, None] * b[None, :]
        else:
            x += b[None, :]
        # Native HC combine stores BF16 before the publisher's HC mean.
        x = x.to(residual.dtype.element_ty).to(tl.float32)
    mean = tl.sum(x, axis=0) / HC
    tl.store(out + row * H + col, mean, mask=col < H)


def _feature_impl(residual: torch.Tensor, block: torch.Tensor | None,
                  injection: torch.Tensor | None, hc_count: int) -> torch.Tensor:
    rows, width = residual.shape
    h = width // hc_count
    if not residual.is_cuda:
        state = residual.unflatten(-1, (hc_count, h))
        if block is not None:
            if injection is None:
                state = (state.float() + block.float()[:, None, :]).to(residual.dtype)
            else:
                gain = 2 * torch.sigmoid(injection.float() / hc_count)
                state = (state.float() + gain[:, :, None] * block.float()[:, None, :]).to(residual.dtype)
        return state.mean(-2)
    out = residual.new_empty((rows, h))
    if rows:
        _completed_feature[(rows, triton.cdiv(h, 256))](
            residual, block, injection, out, residual.stride(0),
            block.stride(0) if block is not None else 0,
            injection.stride(0) if injection is not None else 0,
            H=h, HC=hc_count, TILE=256, HAS_BLOCK=block is not None,
            HAS_INJ=injection is not None, num_warps=4, enable_fp_fusion=False)
    return out


def _feature_fake(residual: torch.Tensor, block: torch.Tensor | None,
                  injection: torch.Tensor | None, hc_count: int) -> torch.Tensor:
    return residual.new_empty((residual.shape[0], residual.shape[1] // hc_count))


direct_register_custom_op(op_name='thinkingcap_hc_feature', op_func=_feature_impl,
                          fake_impl=_feature_fake, tags=(torch.Tag.needs_fixed_stride_order,))


def completed_hc_feature(residual, block, injection, hc_count):
    return torch.ops.vllm.thinkingcap_hc_feature(residual, block, injection, hc_count)


def promote_draft_cache_spec(spec):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, replace_as
    if not isinstance(spec, SlidingWindowSpec):
        raise ValueError('Expected a sliding-window draft cache before allocation promotion')
    return replace_as(spec, FullAttentionSpec, drop=('extra_retained_tokens',))


def retain_draft_prefix_cache(draft):
    if os.environ.get('THINKINGCAP_DFLASH_RETAIN_PREFIX') != '1':
        return
    for layer in draft.model.layers:
        attn = layer.self_attn.attn
        original = attn.get_kv_cache_spec
        def full_allocation(self, cfg, original=original):
            return promote_draft_cache_spec(original(cfg))
        attn.get_kv_cache_spec = types.MethodType(full_allocation, attn)
    log.info('DFlash2 draft caches use full allocation for fine-grained prefix reuse; attention window remains 2048')


@torch.inference_mode()
def install_draft_quantization(target, draft):
    if os.environ.get('THINKINGCAP_ENABLE_DFLASH2') != '1':
        return
    retain_draft_prefix_cache(draft)
    from thinkingcap_draft_head import install, quantize_head
    from thinkingcap_dense_quant import DualPrecisionMethod
    records = []
    if os.environ.get('THINKINGCAP_DFLASH_DENSE_FP8', '0') == '1':
        # Context KV retains its prebuilt, fused BF16 projection. Queries use
        # adaptive FP8 linears; convolution and selector weights remain BF16.
        for name, module in list(draft.model.named_modules()):
            selected = (name == 'fc' or name.endswith(('.self_attn.qkv_proj', '.self_attn.o_proj',
                                                     '.mlp.gate_up_proj', '.mlp.down_proj')))
            if not selected or not hasattr(module, 'quant_method'):
                continue
            source = module.weight
            original = module.quant_method
            if source.dtype != torch.bfloat16 or getattr(module, 'bias', None) is not None:
                raise ValueError(f'Unexpected draft projection layout at {name}')
            quantized = quantize_head(module, 'fp8')
            probe = torch.randn(8, source.shape[1], device=source.device, dtype=source.dtype)
            reference = original.apply(module, probe).float()
            actual = quantized.quant_method.apply(quantized, probe).float()
            error = float((reference - actual).norm() / reference.norm().clamp_min(1e-12))
            if error >= .05:
                raise ValueError(f'Draft FP8 probe failed at {name}: {error}')
            module.quant_method = DualPrecisionMethod(original, quantized, max_rows=128)
            assert module.weight is source
            records.append({'name': name, 'shape': list(source.shape), 'relative_error': error})
    install(target, draft)
    receipt = {'backbone': records, 'target_unchanged': True,
               'context_kv': 'original fused BF16', 'large_batch_fc': 'original BF16',
               'head': os.environ.get('THINKINGCAP_DRAFT_HEAD', 'bf16')}
    draft._thinkingcap_dflash_quant = receipt
    cache = Path(os.environ.get('THINKINGCAP_RECEIPT_DIR', '/cache'))
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'thinkingcap-dflash-quant.json').write_text(json.dumps(receipt, indent=2) + '\n')
    log.info('ThinkingCap DFlash2 private quantization installed: %d projections', len(records))
