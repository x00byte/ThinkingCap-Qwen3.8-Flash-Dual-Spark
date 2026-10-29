# Setup

Run these commands from the repository root. Prepare both hosts before moving production traffic. The scripts do not manage your existing model selector.

## Download weights

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) if needed. On each host:

```bash
hf download bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4 \
  --revision df3922b21fc28a8604cc9b31451495690d155194 \
  --local-dir /srv/models/bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4
hf download incoai/Qwen3.8-Flash-Next-DFlash2 \
  --revision a22a1744e85c17364dcc54a6a420af7135d4c7f6 \
  --local-dir /srv/models/incoai/Qwen3.8-Flash-Next-DFlash2
python3 models.py --root /srv/models
```

`models.py` checks the pinned file sizes and SHA256 or Git blob hashes. You can instead copy a verified download to the second host and run the same verification there. Weight verification reads approximately 133 GB per host.

The target’s Qwen Community License and the draft model’s Apache 2.0 terms are available from their model repositories. The model cards contain sampling guidance; the server profile does not enforce one client sampling policy.

## Build the runtime

On each host:

```bash
python3 runtime/verify_overlay.py
# This builds a CPU-only source overlay; it does not start inference.
docker build -t thinkingcap-qwen38:20261029 runtime
# CPU-only CLI and dependency check; no model is loaded.
docker run --rm --network none --entrypoint python3 \
  -v "$PWD:/recipe:ro" thinkingcap-qwen38:20261029 /recipe/runtime/check_contract.py
```

The base is pinned by digest. Source files are exported from the running image and recorded in `runtime/overlay-lock.json`. Building this recipe does not reproduce the original Docker image ID: layer structure and build metadata differ. [Runtime notes](../runtime/README.md) record provenance and the limits of packaging validation.

## Configure each host

```bash
cp config.example.json config.local.json
```

Set `head_addr` and `worker_addr` to the two hosts’ fabric IPv4 addresses. The example uses documentation-only addresses and must be edited. Set `fabric_interface`, `fabric_hca` and `gid_index` from your working RoCE configuration. Interface and HCA names can differ between hosts; each host reads its own local file.

Check that the selected GID is active and IPv4 RoCE v2, and that the two hosts can reach each other over the fabric. The launcher passes the configured values to NCCL; it does not discover or repair the network.

Set `model_dir`, `draft_dir`, `cache_dir`, and `cpuset` for each machine. Use a dedicated persistent cache directory. The default `image` matches the build command above. Only `thinkingcap` is supported by this packaged runtime.

Both ranks must use the same target and draft revisions, image contents, sequence/context settings, and master port. The API listens on port 8888 by default. Host networking exposes it on all interfaces; keep access within your trusted network. No API authentication is configured by this recipe.

## Start

Drain traffic and stop the current GPU service through its own maintenance procedure. Verify weights on both hosts before the first launch.

On the worker:

```bash
python3 serve.py plan --rank 1
python3 serve.py start --rank 1
```

On the head:

```bash
python3 serve.py plan --rank 0
python3 serve.py start --rank 0
curl --fail http://localhost:8888/health
curl --fail http://localhost:8888/v1/models
```

`start` refuses an existing container of the configured name, active GPU compute processes or an occupied rank-local listen port. It checks image identity labels, model file sizes and required paths. It creates only its own container and never evicts another service. Rank 1 is headless; container creation does not prove API readiness. Startup can take several minutes.

Inspect each host’s logs if startup fails:

```bash
python3 serve.py status
# Container name comes from config.local.json.
docker logs --tail 100 thinkingcap-qwen38
```

After `/health` succeeds, try a small request:

```bash
curl --fail http://localhost:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"thinkingcap-qwen38","messages":[{"role":"user","content":"Reply with READY."}],"max_tokens":256,"temperature":0.6,"chat_template_kwargs":{"enable_thinking":false}}'
```

For agent clients, use `http://HEAD_ADDRESS:8888/v1` and model `thinkingcap-qwen38`. Increase load gradually and inspect memory headroom, cache reuse, tool calls and task completion before relying on sixteen concurrent streams. This publication did not run new GPU or load tests.

## Stop and recover

Run on both hosts:

```bash
python3 serve.py stop
```

This removes only the configured container with this project’s ownership label. It preserves weights and cache. A failed startup leaves its container available for log inspection; stop it before retrying. Restore another service with that service’s own launcher.

For local checks:

```bash
python3 -m unittest discover -s tests -v
```

## Credits

The setup uses the [myllmbox cluster runtime](https://github.com/myllmbox/qwen38-flash-next-cluster-recipe), [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), and the model/runtime contributions credited in the [README](../README.md#credits).
