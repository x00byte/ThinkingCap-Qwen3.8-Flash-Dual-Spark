"""Private quantized MTP vocabulary head; the shared target head stays intact."""
from __future__ import annotations

import copy
import json
import os
import torch
from torch import nn
from vllm.logger import init_logger

log = init_logger("vllm.thinkingcap_draft_head")


class FP8HeadMethod:
    def __init__(self, n, k):
        self.n, self.k = n, k

    def process_weights_after_loading(self, layer):
        return

    def apply(self, layer, x, bias=None):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear
        return apply_fp8_marlin_linear(x, layer.weight, layer.weight_scale, None,
                                      self.n, self.k, bias)


class FP4HeadMethod(FP8HeadMethod):
    def apply(self, layer, x, bias=None):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
        return apply_fp4_marlin_linear(x, layer.weight, layer.weight_scale,
                                      layer.weight_global_scale, bias, self.n, self.k)


class FP4RefinedHeadMethod(FP4HeadMethod):
    """Use FP4 to shortlist rows, then score those rows with the original BF16 head."""
    def apply(self, layer, x, bias=None):
        from mbx_draft_head2 import _cand_logits_dense_kernel
        from vllm.triton_utils import triton
        scores = super().apply(layer, x, bias)
        flat = scores.reshape(-1, self.n)
        hidden = x.reshape(-1, self.k).to(torch.bfloat16).contiguous()
        count = min(128, self.n)
        candidates = flat.topk(count, dim=-1, sorted=False).indices.to(torch.int32).contiguous()
        exact = torch.empty((hidden.shape[0], count), device=x.device, dtype=torch.float32)
        if hidden.shape[0] > 0:
            reference = layer._draft_reference_weight
            _cand_logits_dense_kernel[(hidden.shape[0], triton.cdiv(count, 32))](
                hidden, candidates, reference, exact, count,
                hidden.stride(0), candidates.stride(0), reference.stride(0), exact.stride(0),
                H=self.k, BLOCK_C=32, BLOCK_H=128)
        # This is the draft proposal distribution. vLLM's verifier retains the
        # target distribution, including target tokens outside the shortlist.
        output = torch.full_like(flat, float('-inf'))
        output.scatter_(1, candidates.long(), exact.to(output.dtype))
        return output.reshape(scores.shape)


@torch.inference_mode()
def quantize_head(original, fmt='fp8'):
    weight = original.weight
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        raise ValueError('Expected a plain BF16 target vocabulary shard')
    if getattr(original, 'bias', None) is not None:
        raise ValueError('Only bias-free vocabulary heads are supported')
    n, k = weight.shape
    source_ptr = weight.data_ptr()
    # nn.Module shallow copies share their parameter registry. Detach each registry
    # before replacing any parameter; copying only the module would alter the target.
    head = copy.copy(original)
    head._parameters = dict(original._parameters)
    head._buffers = dict(original._buffers)
    head._modules = dict(original._modules)
    head._non_persistent_buffers_set = set(original._non_persistent_buffers_set)
    head.output_size_per_partition, head.input_size_per_partition = n, k
    chunk = 2048
    if fmt == 'fp8':
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
        quant = torch.empty((n, k), device=weight.device, dtype=torch.float8_e4m3fn)
        scales = torch.empty(n, device=weight.device, dtype=torch.float32)
        for a in range(0, n, chunk):
            block = weight[a:a + chunk].float()
            scale = (block.abs().amax(1) / 448.0).clamp(min=1e-12)
            quant[a:a + chunk] = (block / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
            scales[a:a + chunk] = scale
        head.weight = nn.Parameter(quant, requires_grad=False)
        head.weight_scale = nn.Parameter(scales, requires_grad=False)
        head.orig_dtype = torch.bfloat16
        prepare_fp8_layer_for_marlin(head, size_k_first=False)
        head.quant_method = FP8HeadMethod(n, k)
    elif fmt in ('nvfp4', 'nvfp4-refine'):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
        quant = torch.empty((n, k // 2), device=weight.device, dtype=torch.uint8)
        scales = torch.empty((n, k // 16), device=weight.device, dtype=torch.float8_e4m3fn)
        global_scale = (weight.abs().amax().float() / (448.0 * 6.0)).clamp(min=1e-12)
        mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=weight.device)
        for a in range(0, n, chunk):
            block = weight[a:a + chunk].float().view(-1, k // 16, 16)
            scale = (block.abs().amax(-1) / (6.0 * global_scale)).clamp(min=2 ** -9, max=448).to(torch.float8_e4m3fn)
            values = block / (scale.float() * global_scale).unsqueeze(-1)
            code = torch.bucketize(values.abs().clamp(max=6), mid).to(torch.uint8)
            code = (code | (((values < 0) & (code > 0)).to(torch.uint8) << 3)).reshape(-1, k)
            quant[a:a + chunk] = code[:, 0::2] | (code[:, 1::2] << 4)
            scales[a:a + chunk] = scale
        head.weight = nn.Parameter(quant, requires_grad=False)
        head.weight_scale = nn.Parameter(scales, requires_grad=False)
        head.weight_global_scale = nn.Parameter(global_scale, requires_grad=False)
        head.params_dtype = torch.bfloat16
        prepare_fp4_layer_for_marlin(head)
        head.quant_method = FP4HeadMethod(n, k)
        if fmt == 'nvfp4-refine':
            head.register_buffer('_draft_reference_weight', weight.detach(), persistent=False)
            head.quant_method = FP4RefinedHeadMethod(n, k)
    else:
        raise ValueError('Unknown draft head format: ' + fmt)
    assert original.weight is weight and weight.data_ptr() == source_ptr
    assert weight.dtype == torch.bfloat16 and tuple(weight.shape) == (n, k)
    assert head._parameters is not original._parameters
    assert head.weight.data_ptr() != source_ptr
    torch.cuda.empty_cache()
    return head


@torch.inference_mode()
def install(target_model, draft_model):
    fmt = os.environ.get('THINKINGCAP_DRAFT_HEAD', '')
    if not fmt:
        return
    original = draft_model.lm_head
    target_refs = [name for name, module in target_model.named_modules() if module is original]
    if not target_refs:
        raise RuntimeError('Expected the target and draft to share their original vocabulary head')
    head = quantize_head(original, fmt)
    probe = torch.randn(8, original.weight.shape[1], dtype=torch.bfloat16, device=original.weight.device)
    reference = original.quant_method.apply(original, probe, bias=None).float()
    actual = head.quant_method.apply(head, probe, bias=None).float()
    mask = torch.isfinite(actual)
    relative_error = float((reference[mask] - actual[mask]).norm() / reference[mask].norm().clamp(min=1e-12))
    top1_agreement = float((reference.argmax(-1) == actual.argmax(-1)).float().mean())
    limit = 0.05 if fmt == 'fp8' else 0.15
    if not relative_error < limit:
        raise RuntimeError(f'Draft head quantization error {relative_error:.4f} exceeds {limit}')
    draft_model.lm_head = head
    assert all(dict(target_model.named_modules())[name] is original for name in target_refs)
    receipt = {'format': fmt, 'shape': list(original.weight.shape),
               'relative_error': relative_error, 'top1_agreement': top1_agreement,
               'target_head_unchanged': True,
               'target_references': target_refs}
    log.info('ThinkingCap private draft head installed: %s', json.dumps(receipt))
