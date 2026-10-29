"""Bounded, history-only lookup proposals with exact sparse-cache bookkeeping.

The override decision never depends on sampled model drafts. A full draft block
is replaced together, so no later cached conditional depends on a replaced
predecessor. Every overridden q row becomes the matching point mass, and the
candidate cache remembers that row for the following step's clearing operation.
"""
import os
import torch
from vllm.triton_utils import tl, triton
from vllm.logger import init_logger
from thinkingcap_hyperqwen_lookup import suffix_lookup

log = init_logger('vllm.thinkingcap_lookup')


@triton.jit
def _override(draft, draft_stride, lookup, lookup_stride, match, valid, mapping,
              cached_ids, first_scores, logits, q_stride0: tl.constexpr, q_stride1: tl.constexpr,
              hits, used, N: tl.constexpr, K: tl.constexpr, TOPK: tl.constexpr,
              MIN_MATCH: tl.constexpr, HAS_Q: tl.constexpr,
              BLOCK: tl.constexpr):
    row = tl.program_id(0)
    state = tl.load(mapping + row)
    take = (state >= 0) & (tl.load(match + row) >= MIN_MATCH) & (tl.load(valid + row) >= K)
    cols = tl.arange(0, BLOCK)
    # The first-position selector distribution is computed before any draft
    # sample is drawn (previous=0 in the native selector walk). Its argmax is
    # a deterministic corroboration signal. Later-position scores depend on
    # sampled predecessors and MUST NOT be used to choose an override branch.
    if take:
        first_token = tl.load(lookup + row * lookup_stride).to(tl.int64)
        first_ids = tl.load(cached_ids + state * K * TOPK + cols, cols < TOPK, other=-1)
        scores = tl.load(first_scores + row * K * TOPK + cols, cols < TOPK, other=-float('inf'))
        best = tl.max(scores)
        corroborated = tl.sum(((first_ids == first_token) & (scores == best) & (cols < TOPK)).to(tl.int32)) > 0
        take = take & corroborated & (best > -float('inf'))
    tl.store(used + row, take.to(tl.int32))
    if not take:
        return
    for step in range(K):
        token = tl.load(lookup + row * lookup_stride + step).to(tl.int64)
        tl.store(draft + row * draft_stride + step, token)
        if HAS_Q:
            cache_base = (state * K + step) * TOPK
            previous = tl.load(cached_ids + cache_base + cols, cols < TOPK, other=0)
            q_base = logits + state * q_stride0 + step * q_stride1
            # Avoid an unequal write race if the lookup token is already a
            # candidate: both stores for that token then write exactly zero.
            tl.store(q_base + previous, tl.where(previous == token, 0., -float('inf')),
                     cols < TOPK)
            tl.store(q_base + token, 0.)
            # All previous support has been cleared. Remember the new token
            # in slot zero so the NEXT model proposal clears it as well.
            tl.store(cached_ids + cache_base, token)
    tl.atomic_add(hits, 1)


class SelectiveLookup:
    def __init__(self, speculator, req_states, window=32768, min_match=8):
        if not 8 <= min_match <= 32:
            raise ValueError('Lookup match threshold must be in [8,32]')
        if not 1024 <= window <= 131072:
            raise ValueError('Bound lookup scans to [1024,131072] tokens')
        self.req_states = req_states
        self.k = speculator.num_speculative_steps
        self.window, self.min_match = window, min_match
        n, dev = speculator.max_num_reqs, speculator.device
        self.tokens = torch.zeros(n, self.k, dtype=torch.int32, device=dev)
        self.match = torch.zeros(n, dtype=torch.int32, device=dev)
        self.valid = torch.zeros_like(self.match)
        self.used = torch.zeros_like(self.match)
        self.hits = torch.zeros((), dtype=torch.int64, device=dev)
        self._steps = 0
        self._rows = 0
        self._report_every = int(os.environ.get('THINKINGCAP_LOOKUP_REPORT_STEPS', '256'))
        self._report_event = None
        self._report_buffer = torch.empty((), dtype=torch.int64, device='cpu', pin_memory=True)
        self._report_rows = 0

    def apply(self, speculator, batch, num_sampled, output):
        n = batch.num_reqs
        if not n:
            return
        mapping = torch.where(num_sampled[:n] > 0, batch.idx_mapping[:n], -1).contiguous()
        self.match.zero_()
        self.valid.zero_()
        suffix_lookup(self.req_states.all_token_ids.gpu, self.req_states.total_len.gpu,
                      mapping, n, self.k, nmin=self.min_match, nmax=32,
                      search_max=self.window, out_tokens=self.tokens,
                      out_len=self.match, out_valid=self.valid)
        q = speculator.draft_logits
        _override[(n,)](output, output.stride(0), self.tokens, self.tokens.stride(0),
                        self.match, self.valid, mapping,
                        speculator.candidate_sampler.cached_candidate_ids,
                        speculator.candidate_sampler.scores, q,
                        q.stride(0) if q is not None else 0,
                        q.stride(1) if q is not None else 0,
                        self.hits, self.used, N=n, K=self.k,
                        TOPK=speculator.candidate_sampler.top_k,
                        MIN_MATCH=self.min_match, HAS_Q=q is not None,
                        BLOCK=triton.next_power_of_2(speculator.candidate_sampler.top_k),
                        num_warps=1)
        self._steps += 1
        self._rows += n
        if self._report_every > 0 and self._steps % self._report_every == 0:
            if self._report_event is not None:
                if not self._report_event.query():
                    return
                log.info('ThinkingCap lookup cumulative: blocks_overridden=%d, proposal_rows=%d',
                         int(self._report_buffer.item()), self._report_rows)
            self._report_buffer.copy_(self.hits, non_blocking=True)
            self._report_event = torch.cuda.Event()
            self._report_event.record()
            self._report_rows = self._rows


def attach_lookup(speculator, req_states):
    if os.environ.get('THINKINGCAP_SELECTIVE_LOOKUP', '0') != '1':
        return
    if speculator.enable_adaptive_verification:
        raise ValueError('Lookup qualification requires adaptive verification disabled')
    if speculator.draft_watermarker is not None:
        raise ValueError('Lookup watermark interaction is not qualified')
    if speculator.draft_logits is None:
        raise ValueError('Initial lookup bridge requires probabilistic draft-cache bookkeeping')
    window = int(os.environ.get('THINKINGCAP_LOOKUP_WINDOW', '32768'))
    minimum = int(os.environ.get('THINKINGCAP_LOOKUP_MIN_MATCH', '8'))
    speculator.thinkingcap_lookup = SelectiveLookup(speculator, req_states, window, minimum)
    log.info('ThinkingCap lookup: deterministic first-position top1 corroboration; full %d-token blocks, min match %d, window %d',
             speculator.num_speculative_steps, minimum, window)
