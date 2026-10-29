# Measurement notes

These results come from x00byte’s high concurrency agentic research workflow on two Sparks, comparing October 21–24 with October 26–28, 2026. ThinkingCap was used throughout the second period. This repository contains the operator’s aggregate figures; raw research prompts, findings and run logs remain private.

The median generated-token reduction is `1 − 22,894 / 46,116 = 50.36%`. The median-duration reduction is `1 − 23.5 / 48.7 = 51.75%`. The README rounds the underlying ratios to one decimal place.

The approximately 1.86× completed-runs-per-worker-hour figure uses summed run durations across concurrent workers. It was supplied with the workflow results and cannot be reconstructed from median duration. Exact worker-hour totals were not included in the aggregate summary.

Generated tokens include reasoning and visible output. Repeated usage events were deduplicated; missing usage was excluded rather than treated as zero. Input tokens served include cached context, so that metric includes tokens that did not require fresh prefill computation.

The two periods were not a matched-task experiment. Task mix and context reuse may affect the observed gains.

Actionable Medium/High/Critical findings were 7 versus 5, with rates of 1.05 versus 1.03 per 100 worker-hours. That small sample does not establish a quality improvement. The practical gains observed here were lower token use and faster research turnover, with a nearly unchanged findings rate.

## Serving measurements

The [October 25 serving data](../benchmarks/serving-2026-10-25.json) contains the recorded prefill and aggregate prose results. It preserves the exact numeric values and SHA256 identifiers for the source report and its reused core measurements. The core samples are counted once.

Prose tests used one batch each at concurrency 1, 4 and 16. Every request generated exactly 400 tokens, with temperature 0, thinking disabled, `min_tokens=400` and `ignore_eos=true`. The prompt requested a long story about repairing an observatory during a winter storm. The server reported no running or queued requests before each batch, and no external completions during it.

Aggregate prose decode is total completion tokens divided by the interval from the earliest first streamed token to the last response ending. End-to-end output is total completion tokens divided by the entire batch’s wall time. These are aggregate rates across all streams, not per-stream rates or averages of per-request rates.

Prefill used fixed-seed random-word prompts with unique markers, followed immediately by an identical warm request. Each request allowed one output token. Cold throughput is prompt tokens divided by time to first token, so it includes request and first-token overhead. Both cold requests reported zero cached tokens; warm requests reused 10,224 of 10,282 and 40,992 of 41,044 prompt tokens respectively. Warm TTFT measures cached reuse and should not be interpreted as cold prefill throughput.

These are archived serving measurements, separate from the October 26–28 research-run aggregates. No new inference or GPU benchmark was run for this publication update.

## Credits

Measurements and workflow context: x00byte. Model and runtime contributors are credited in the [README](../README.md#credits).
