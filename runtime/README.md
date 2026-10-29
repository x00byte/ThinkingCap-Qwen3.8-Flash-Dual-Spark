# Runtime source

The Dockerfile uses the public myllmbox cluster image at digest `sha256:a87154e4edca649013b7b8934df70ac53e9f096f5d0e42a823f0ee36452ca315`. It adds the source overlay exported from the running ThinkingCap deployment on October 29, 2026. The private machine image ID in `source-lock.json` identifies the source snapshot; it is not a downloadable dependency.

The overlay contains sixteen changed vLLM files and five ThinkingCap modules. All thirteen vLLM shared-library hashes matched the public base, and the base already contains identical B12X source and the PLE checkpoint helper. MBX’s quantization module stays in the upstream image; the small included patch adds the local dense-precision hook.

The overlay supplies the model/DFlash feature bridge, refined draft head, adaptive dense precision, selective history lookup, draft cache handling, local prefill cadence, QSA workspace accounting, and compatible distributed/kernel calls. GDN/PLE checkpoint support comes from the pinned base. The profile configures native 262,144-token context and sixteen sequences.

The exported sources receive modification/attribution comments; their original hashes are also recorded in `source-lock.json`. `overlay-lock.json` pins every shipped overlay file. The Docker build checks those hashes and the patched quantization module. The image label records the overlay fingerprint, which the launcher checks before starting. It is a packaging identity check, not cryptographic attestation of an arbitrary third-party image.

Source provenance includes myllmbox’s [cluster recipe at 99e8864](https://github.com/myllmbox/qwen38-flash-next-cluster-recipe/tree/99e88640508d709959a45740ed93cc8c5887477e), [vLLM GDN checkpoints](https://github.com/vllm-project/vllm/pull/60050), and [QSA workspace accounting](https://github.com/vllm-project/vllm/pull/60490). The included vLLM files carry deployment modifications; they are not an unmodified upstream release. See [third-party notices](../THIRD_PARTY.md).

The portable image is a new packaging of the serving source. It has not been validated by launching a GPU model from this repository. Existing operator records cover live startup, native context, a completion, and repeated-prefix cache hits on the original deployment. They do not establish a comprehensive quality or performance result for a fresh installation.

## Credits

[myllmbox](https://github.com/myllmbox/qwen38-flash-next-cluster-recipe), [vLLM](https://github.com/vllm-project/vllm), [B12X](https://github.com/local-inference-lab/b12x), and [HyperQwen](https://github.com/syv-ai/HyperQwen) supply the runtime foundations and adapted work described above.
