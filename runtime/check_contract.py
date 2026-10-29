"""Exercise the installed CLI and dependency interfaces without a GPU engine."""
from pathlib import Path
import inspect
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve
from b12x.comm import roce
from b12x.gemm.blockscaled._ops import linear
import thinkingcap_dflash2
import thinkingcap_draft_head
import thinkingcap_dense_quant
import thinkingcap_lookup
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform

# Set parser defaults only. No engine, CUDA context or model is constructed.
vllm.platforms._current_platform = CpuPlatform()
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.cli.serve import make_arg_parser
from vllm.engine.arg_utils import AsyncEngineArgs

assert roce.API_VERSION == 1
assert 'mode' in inspect.signature(linear).parameters
cfg = serve.load(serve.ROOT / 'config.example.json')
for rank in (0, 1):
    command = serve.command(cfg, rank)
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(command[command.index('serve') + 1:])
    engine = AsyncEngineArgs.from_cli_args(args)
    assert engine.max_num_seqs == 16 and engine.max_model_len == 262144
    assert engine.prefix_match_unit == 64 and engine.block_size == 3840
    assert engine.mamba_ssm_cache_dtype == 'float32'
    assert not engine.hf_overrides
print(json.dumps({'passed': True, 'ranks': 2,
                  'scope': 'CLI parsing and dependency imports; no GPU engine or inference'}))
