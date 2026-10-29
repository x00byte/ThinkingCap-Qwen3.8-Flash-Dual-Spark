"""Use FP8 for small projection batches and retain native BF16 for prefill."""
import json
from pathlib import Path
import torch
from vllm.logger import init_logger

log = init_logger('vllm.thinkingcap_dense_quant')


class DualPrecisionMethod:
    def __init__(self, original_method, quantized, max_rows=128):
        self.original_method = original_method
        self.quantized = quantized
        self.max_rows = max_rows

    def process_weights_after_loading(self, layer):
        return

    def apply(self, layer, x, bias=None):
        rows = x.numel() // x.shape[-1]
        if rows <= self.max_rows:
            return self.quantized.quant_method.apply(self.quantized, x, bias)
        return self.original_method.apply(layer, x, bias)


@torch.inference_mode()
def install(model, include):
    from mbx_opt_quant import _selected, _EXCLUDE, _EXCLUDE_SUFFIX
    from thinkingcap_draft_head import quantize_head
    records = []
    extra_bytes = 0
    for name, module in _selected(model, include, _EXCLUDE, _EXCLUDE_SUFFIX, 1 << 20):
        if isinstance(module.quant_method, DualPrecisionMethod) or getattr(module, 'bias', None) is not None:
            continue
        original_method = module.quant_method
        original_weight = module.weight
        quantized = quantize_head(module, 'fp8')
        x = torch.randn(8, original_weight.shape[1], device=original_weight.device, dtype=torch.bfloat16)
        reference = original_method.apply(module, x).float()
        actual = quantized.quant_method.apply(quantized, x).float()
        error = float((reference - actual).norm() / reference.norm().clamp(min=1e-12))
        if not error < 0.05:
            log.warning('Keeping BF16 projection %s: FP8 relative error %.4f', name, error)
            continue
        assert module.weight is original_weight and module.weight.dtype == torch.bfloat16
        module.quant_method = DualPrecisionMethod(original_method, quantized)
        added = sum(p.numel() * p.element_size() for p in quantized.parameters())
        extra_bytes += added
        records.append({'name': name, 'shape': list(original_weight.shape),
                        'relative_error': error, 'extra_bytes': added})
    if not records:
        raise RuntimeError("No eligible ThinkingCap dense projections found")
    torch.cuda.empty_cache()
    receipt = {'mode': 'FP8 Marlin for <=128 rows; original BF16 for larger batches',
               'target_bf16_parameters_retained': True, 'extra_bytes': extra_bytes,
               'projections': records}
    model._thinkingcap_dual_precision = receipt
    Path('/cache/thinkingcap-target-dense-fp8.json').write_text(json.dumps(receipt, indent=2) + '\n')
    log.info('ThinkingCap dual precision: %d projections, %.1f MiB extra; native BF16 prefill retained',
             len(records), extra_bytes / 2**20)
    return extra_bytes
