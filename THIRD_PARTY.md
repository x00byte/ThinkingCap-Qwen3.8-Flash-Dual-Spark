# Third-party notices

The profile settings, orchestration and workload report are maintained by x00byte. No public license has been selected for x00byte’s original contributions. This does not replace any third-party license.

## Included source

- `runtime/overlay/vllm/`: adapted vLLM source, Apache 2.0. Existing SPDX and copyright notices are retained. Files contain local serving modifications and myllmbox integration; their exact content is pinned in `runtime/overlay-lock.json`. License: `runtime/licenses/LICENSE.vLLM`. Source: [vLLM](https://github.com/vllm-project/vllm), [Local Inference Lab vLLM](https://github.com/local-inference-lab/vllm), and the [myllmbox cluster runtime](https://github.com/myllmbox/qwen38-flash-next-cluster-recipe).
- `runtime/overlay/thinkingcap_hyperqwen_lookup.py`: lookup-augmented drafting kernel adapted from [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen), Apache 2.0. License: `runtime/licenses/LICENSE.HyperQwen`. The surrounding `thinkingcap_lookup.py` adapts it to the deployed DFlash probability-cache and rejection-sampling path.
- The remaining `thinkingcap_*.py` modules implement local target/draft integration, quantized draft-head refinement and adaptive dense precision, using the runtime’s existing vLLM/MBX interfaces. MBX’s `mbx_opt_quant.py` remains in the upstream image; the repository contains only the local hook patch.

## Runtime dependencies

[myllmbox](https://github.com/myllmbox/qwen38-flash-next-cluster-recipe) provides the pinned public image, compiled MBX components and PLE checkpoint helper. Those components are not reimplemented or redistributed separately here. The upstream image’s terms and notices remain applicable.

[B12X](https://github.com/local-inference-lab/b12x) provides Blackwell kernels and RoCE communication, under Apache 2.0. Its source is supplied by the upstream image rather than duplicated in this repository. Its license is preserved at `runtime/licenses/LICENSE.B12X`. CUDA, PyTorch, FlashInfer, CUTLASS and other installed dependencies retain their upstream terms.

## Models

Model weights are downloaded from their publishers and are not bundled:

- [BottleCap AI ThinkingCap Qwen3.8 Flash Next NVFP4](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4): the ThinkingCap fine-tune and quantization of the Qwen base model, under the Qwen Community License 1.0. Consult the publisher’s license for the applicable conditions.
- [Qwen3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): base architecture, model and tokenizer.
- [Inco AI Qwen3.8 Flash Next DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2): Apache 2.0 draft model. [DFlash](https://github.com/z-lab/dflash) supplies the speculative-decoding research lineage.

## Artwork

The README cover was produced with the built-in image-generation tool for this project. The exact prompt is in `assets/image-prompt.txt`. It is an illustration of the setup and does not imply endorsement by any hardware or model publisher.

## Credits

Credit belongs to the model publishers and software contributors linked above. x00byte’s workload measurements describe a personal deployment, not results endorsed by those projects.
