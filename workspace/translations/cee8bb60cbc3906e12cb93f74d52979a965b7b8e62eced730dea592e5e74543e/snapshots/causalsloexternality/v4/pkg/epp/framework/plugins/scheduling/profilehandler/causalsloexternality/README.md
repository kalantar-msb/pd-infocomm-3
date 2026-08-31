# Causal SLO Externality Profile Handler and Picker

Joint (decode endpoint, prefill placement) routing for P/D disaggregation. Prices each
candidate by the smooth SLO value it destroys among an endpoint's residents, evaluated
causally, and picks the argmin over the full cross product.

## Contents

- [What it does](#what-it-does)
- [Plugins](#plugins)
  - [CausalSLOExternalityHandler](#causalsloexternalityhandler)
  - [CausalSLOExternalityPicker](#causalsloexternalitypicker)
- [Why two plugins](#why-two-plugins)
- [Why a picker and not a scorer](#why-a-picker-and-not-a-scorer)
- [Configuration](#configuration)
- [Runtime requirements](#runtime-requirements)
- [Observability](#observability)
- [Limitations](#limitations)

---

## What it does

The decision rule is a joint argmin over decode endpoint `d` and prefill placement `p`:

```
J(d, p) = V * ( externality(d, p) - ownGood(d, p) )
```

`externality` is the SLO value the placement destroys among the residents already on the
candidate endpoints. `ownGood` is the value the placement earns for the arriving request.
`V` is a common positive multiplier.

**What differs between candidates** is the SLO value already banked in a candidate's
residents. Two endpoints with identical queue depth and identical KV occupancy are not
interchangeable: one may hold residents sitting just inside their end-to-end deadline,
where one extra co-scheduled prefill flips them to a miss; the other may hold residents
with comfortable slack, or residents already past their deadline, where the same prefill
costs nothing recoverable. Queue depth, KV utilization, and projected TTFT cannot see that
difference, because it is a property of the residents' deadlines rather than of the
endpoint's load.

The pricing is **causal**: every completion model is gated on an admission window, so a
resident that finishes before the arrival is admitted contributes exactly zero. That
gating is what makes the term an externality rather than another load proxy. Heterogeneous
fleets give it the most room, because per-GPU coefficients make the same resident set cost
different amounts on different hardware.

Candidate enumeration is D local candidates plus D x P disaggregated candidates, scored on
one scale and compared in one argmin. The decode endpoint is part of the output on **both**
outcomes: a local win still names a decode endpoint, rather than deferring to the
inherited scorer.

---

## Plugins

### CausalSLOExternalityHandler

**Type:** `causal-slo-externality-handler`
**Interfaces**: `scheduling.ProfileHandler`, `requestcontrol.PreRequest`,
`requestcontrol.ResponseBodyProcessor`

Owns the shared state -- the policy, the coefficient table, and the resident shadow table
-- drives the single role-blind profile, splits the picker's two picks into the final
scheduling result, sets the prefill routing header, and advances the shadow table from
response chunks.

#### What it does

1. `Pick` runs one role-blind profile, once.
2. `ProcessResults` splits the picker's two endpoints: the decode pick becomes the primary
   profile result on its own, the prefill pick moves to a separate key. It also records
   the placement in the shadow table.
3. `PreRequest` clears any inherited `x-prefiller-host-port` header, then sets it to the
   chosen prefill pod when the argmin disaggregated.
4. `ResponseBody` advances each resident's `StepsDone` and realized first token from
   `Usage.CompletionTokens`, and on the terminal chunk folds the realized output length
   into the per-class output-length mean.

#### Inputs consumed

- `TokenizedPrompt` -- required. The prompt token count `a_r`, from a `token-producer`.
- `PrefixCacheMatchInfo` -- optional, per endpoint. The cached prefix, from
  `approx-prefix-cache-producer`. Both are bound by producer name.

### CausalSLOExternalityPicker

**Type:** `causal-slo-externality-picker`
**Interfaces**: `scheduling.Filter`, `scheduling.Picker`

Builds the routing view, rejects endpoints that cannot be priced, and runs the argmin.

#### What it does

`Filter` reads the request once and attaches the per-request operands to each accepted
endpoint: the prompt token count, the SLO class, the decision instant, the per-endpoint
uncached suffix `a_p`, the role split, and each endpoint's routing snapshot. It returns
**both pools**, because the joint argmin needs the full cross product in one picker run.

`Pick` runs the argmin and returns the decode pick at `TargetEndpoints[0]` and, when the
argmin chose to disaggregate, the prefill pick at `TargetEndpoints[1]`.

An endpoint is rejected, and the rejection counted by reason, when its metadata is
missing, its role label is absent or unrecognized, its GPU-type label is absent or not a
key of `coeffsByGpuType`, its metrics are unavailable, or its scraped `CacheBlockSize`
disagrees with the configured `engine.blockSize`.

When the prompt token count is unavailable the picker computes no objective, but the
Filter still validates the candidate set and narrows it to decode-eligible endpoints, and
the picker then returns the lowest-ID one. Both steps are deliberate: nothing downstream of
the picker checks role, and the slice the picker receives is map-ordered, so neither role
safety nor determinism can be assumed here.

---

## Why two plugins

`scheduling.ProfileHandler.Pick` and `scheduling.Picker.Pick` share a method name with
different signatures, so no single Go type can satisfy both interfaces.

The handler is declared **first** and owns the state; the picker looks it up by
`handlerPluginName`. That order is required: plugins are instantiated in configuration
order and registered with the handle only after construction, so by-name references
resolve backward only. A forward reference is a startup error rather than a second,
divergent copy of the shadow table.

State ownership follows lifecycle ownership -- the response hooks that write the table live
on the handler.

`plugin.PluginState` is not used for this link. Its documentation excludes cross-plugin
handoff, and it is keyed per request, whereas the shadow table must outlive every request
in it.

The Filter-to-Picker channel is the per-request endpoint clone: `Picker.Pick` receives no
request, and the director builds a fresh `Endpoint` per scheduling cycle, so an attribute
written in the Filter is readable in the Picker and is private to one request.

**Exactly one `ProfileHandler` is permitted per configuration.** This handler cannot
coexist with `disagg-profile-handler` or `single-profile-handler`, which is why it sets the
prefill header itself rather than delegating.

---

## Why a picker and not a scorer

Two independent reasons:

1. `Scorer.Score` returns `map[Endpoint]float64` -- per endpoint, with no way to name a
   pair, while the objective is over `(decode, prefill placement)`.
2. Every scorer contribution passes through the profile's score-range enforcement, which
   clamps to `[0,1]`, while `J` is signed and unbounded. Clamping does not degrade the
   ranking, it destroys it: every candidate with `J <= 0` collapses to one score.

A picker's own computed objective does not pass through that clamp. The corollary is that
`ScoredEndpoint.Score` carries inherited, already-clamped contributions, and this picker
ignores that field entirely -- it is used only as a tie-break order.

Attach the picker to a profile with **no scorers**.

---

## Configuration

### Handler parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `v` | `float` | Yes | Common multiplier on `externality - ownGood`. |
| `ablation.noCapacity` | `bool` | Yes | Disable the capacity drift term. |
| `ablation.noExternality` | `bool` | Yes | Remove the resident externality term. |
| `ablation.noOwnGood` | `bool` | Yes | Remove the arriving request's own-good term. |
| `ablation.occupancyCapacity` | `bool` | Yes | Use the occupancy-time capacity account. |
| `ablation.decomposed` | `bool` | Yes | Restrict the decode enumeration to one endpoint. See the limitation below -- with no scorers attached this is NOT the load-shaped decode-first control. |
| `activeWorkload` | `string` | Yes | Key of `workloadTargets` to use. |
| `workloadTargets` | `map` | Yes | Per-workload `{tauTtftUs, tauItlUs, tauE2eUs}`. |
| `engine.chunkTokens` | `int` | Yes | Must equal `max_num_batched_tokens`. |
| `engine.blockSize` | `int` | Yes | Must equal `block_size`. Validated against scraped `CacheBlockSize`. |
| `engine.maxBatchSize` | `int` | Yes | Must equal `max_num_seqs`. |
| `signals.gpuTypeLabelKey` | `string` | Yes | Pod label whose value keys `coeffsByGpuType`. |
| `signals.prefixMatchInfoProducerName` | `string` | No | Binds the prefix read by producer name. |
| `signals.tokenizedPromptProducerName` | `string` | No | Binds the token read by producer name. |
| `admissionEstimator` | `string` | Yes | Must be `rollforward`. |
| `transfer.sizeAware` | `bool` | Yes | Use the size-dependent KV transfer price. |
| `transfer.xferBaseUs` | `float` | Yes | Additive base of the size-aware form. |
| `transfer.xferBandwidthGbps` | `float` | If size-aware | Transfer bandwidth. |
| `transfer.kvBytesPerTokenPerGpu` | `float` | If size-aware | KV bytes per token per GPU. |
| `transfer.flatCXferUs` | `int` | If not size-aware | Flat transfer cost. A distinct field from `xferBaseUs`. |
| `coeffsByGpuType` | `map` | Yes | Per-GPU-type `{alphaD, alphaP, c0, c1, cPf, cAttn}`. No fallback entry. |
| `shadowTable.entryTtlSeconds` | `int` | Yes | Reap entries idle this long. |
| `shadowTable.sweepIntervalSeconds` | `int` | Yes | Reaper period. |
| `shadowTable.residentPrefillTokensCap` | `int` | Yes | Caps `S_pf` per occupant and in total. |
| `capacity.tauRefUs` | `int` | If capacity enabled | Reference tau for the capacity scale. |
| `capacity.nomPrefillTokens` | `int` | If capacity enabled | Nominal prefill chunk. |
| `capacity.referenceBatch` | `int` | If occupancy capacity | Fixed decode width. |
| `outputTokenProcessingUs` | `float` | Yes | Per-token post-processing latency, added to every TTFT projection. |

Parameters are decoded with `DisallowUnknownFields`, so a misspelled key is a startup
error. There are deliberately no defaults for the physics: a missing value fails loudly
rather than pricing candidates under invented coefficients.

### Picker parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `handlerPluginName` | `string` | Yes | Name of the handler that owns the shared state. Must be declared earlier. |

### Example

```yaml
plugins:
- type: token-producer
  parameters:
    modelName: meta-llama/Llama-3.3-70B-Instruct
    vllm:
      url: "http://localhost:8000"
- type: approx-prefix-cache-producer
- type: causal-slo-externality-handler
  parameters:
    v: 8
    ablation:
      noCapacity: true
      noExternality: false
      noOwnGood: false
      occupancyCapacity: false
      decomposed: false
    activeWorkload: interactive
    workloadTargets:
      interactive:
        tauTtftUs: 1000000
        tauItlUs: 50000
        tauE2eUs: 16000000
    engine:
      chunkTokens: 2048
      blockSize: 16
      maxBatchSize: 256
    signals:
      gpuTypeLabelKey: pd-infocomm.io/gpu-type
      prefixMatchInfoProducerName: approx-prefix-cache-producer
      tokenizedPromptProducerName: token-producer
    admissionEstimator: rollforward
    transfer:
      sizeAware: true
      xferBaseUs: 50.0
      xferBandwidthGbps: 25.0
      kvBytesPerTokenPerGpu: 81920
      flatCXferUs: 5000
    coeffsByGpuType:
      H100_SXM_80GB:
        alphaD: 16613.537554540144
        alphaP: 16617.85321583337
        c0: 5.347316038602452
        c1: 0.04761401141756073
        cPf: 6.144687138665833
        cAttn: 0.00010075247918809842
    shadowTable:
      entryTtlSeconds: 900
      sweepIntervalSeconds: 30
      residentPrefillTokensCap: 2048
    capacity:
      tauRefUs: 1000000
      nomPrefillTokens: 512
      referenceBatch: 256
    outputTokenProcessingUs: 0.0
- type: causal-slo-externality-picker
  parameters:
    handlerPluginName: causal-slo-externality-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: causal-slo-externality-picker
```

---

## Runtime requirements

Each of these is a setting without which a signal is silently wrong rather than absent.

| Setting | Consequence if omitted |
|---|---|
| `--enable-force-include-usage` on the engine | `Usage.CompletionTokens` arrives only in the final chunk, so every resident's `StepsDone` stays 0 for its whole lifetime and every remaining-steps estimate is wrong, while the shadow table looks present and correct. |
| A reachable tokenizer for the `token-producer` | See the tokenization limitation below. |
| `epp.replicas: 1` | Replicas partition the shadow table, so each replica sees a fraction of the residents and under-counts every externality. |
| One `InferencePool` for all decode endpoints | Two pools mean two EPPs, each blind to the other's endpoints, so no argmin over the full decode set exists. |
| A pod label carrying the GPU type, with `nodeSelector` pinned to the same value | A mismatch serves traffic under the wrong physics. Pinning makes it fail at scheduling time instead. |
| `enable_prefix_caching: true` | No prefix residency to report, so every candidate is priced as cold. |
| A positive `tauTtftUs` and `tauE2eUs` in the selected triple | Enforced at startup. A zero target does not loosen the policy, it flattens it -- see below. |

---

## Observability

Every counter below exists because without it a specific wrong-but-running state is
indistinguishable from correct operation. All carry `plugin_name` and `plugin_type`, so two
configured instances can be compared.

| Metric | What its absence would hide |
|---|---|
| `causal_slo_externality_estimator_substitution_total` | That the closed-form admission estimator is running in place of the scheduler rollout. Non-zero is expected. |
| `causal_slo_externality_tokenization_unavailable_total` | That the arm computed no objective and fell back to the lowest-ID decode-eligible endpoint. |
| `causal_slo_externality_prefix_unobserved_total` | That `a_p` fell back to the full prompt, re-pricing the local/remote boundary. |
| `causal_slo_externality_candidate_rejected_total` (by `reason`) | That a mislabelled endpoint left the candidate set -- otherwise indistinguishable from a routing preference. One reason, `prefill_pick_missing_from_scored_set`, records a disaggregated win that had to run locally. |
| `causal_slo_externality_block_size_mismatch_total` | A config/engine unit disagreement. |
| `causal_slo_externality_shadow_table_entries` | That the resident populations are empty, so every externality is 0 and the mechanism is inert. |
| `causal_slo_externality_shadow_table_reaped_total` | That residents are being charged for requests that are gone. |
| `causal_slo_externality_output_length_folded_total` | The size of the population feeding the per-class `N_out` mean. An unknown share are TRUNCATED counts from requests that never completed, which the response hook cannot distinguish -- see Limitations. |
| `causal_slo_externality_placement_total` | What the request actually ran as, local versus disaggregated. It records the shape of the returned result rather than the argmin's intent, so it cannot disagree with behaviour. |
| `causal_slo_externality_argmin_duration_seconds` | EPP-side work inside the measured TTFT. Grows with D + D x P. |
| `causal_slo_externality_unmapped_gpu_type_at_score_total` | Should be identically zero. Non-zero means an endpoint was priced under zero-valued physics. |

---

## Limitations

- **The scheduler rollout is unreachable at this pin.** The published admission and TTFT
  model replays the engine scheduler over its ordered wait queue. The target exposes
  `Metrics.WaitingQueueSize`, one integer, with no route to the queue contents, the current
  grants, or the step start instant. The closed-form `rollforward` estimator substitutes,
  and is the only accepted value of `admissionEstimator`. Past one batch drain it
  understates admission delay, which over-prices the local candidate and biases toward
  remote prefill. The rollout is implemented and tested against a synthesised snapshot, so
  a future engine patch exporting the queue makes it reachable without re-derivation.

- **An aborted request folds a truncated output length into the `N_out` mean.** The
  specification requires that mean to be updated only on requests that actually completed,
  because a truncated count drags the estimate down. That is not honourable at this pin. The
  response hook carries no error or termination state -- upstream records a TODO to add one
  (`pkg/epp/framework/interface/requestcontrol/plugins.go:76-86`) -- and the server FORCES a
  terminal call with `EndOfStream` true for every request that picked a pod but never
  completed, whether from an error, a client disconnect, or a panic
  (`pkg/epp/handlers/server.go:373-379`). The flag that separates the two,
  `reqCtx.ResponseComplete`, is director-internal and is already set before the hook runs on
  the normal paths (`server.go:537-540`, `:608-610`), so the two calls are identical from
  inside a plugin. Declining to fold is not the conservative alternative: the mean would
  never update and the per-class estimate would stay at its seed of 1 forever -- at which
  point `varDecodeInputs` floors every resident's remaining steps at 1, every resident looks
  one step from completion, the baseline and placed completions land within one iteration of
  each other, and the decode-side externality collapses to ~0 across the whole population,
  degenerating the arm to its own-good term. That is the same degeneration as the
  `tauE2eUs = 0` trap the config refuses to start under. Folding truncated counts biases the
  mechanism; declining to fold disables it. So the fold happens and
  `causal_slo_externality_output_length_folded_total` counts it. **Bias:** an understated
  `N_out` makes residents look closer to finishing, so less value is at risk and the
  externality is under-counted -- toward local placement, the same direction as the
  shadow-table degradations. Read the fold counter against the request total to bound the
  affected share, and note this pushes opposite to the admission-estimator substitution.

- **A silently estimated token count cannot be detected at runtime.** The
  `token-producer`'s `estimate` backend is the zero-config default, and `TokenizedPrompt`
  carries no provenance marker, so an unreachable tokenizer does not surface as an absent
  prompt -- it surfaces as a plausible estimated `a_r` that mis-prices every candidate. The
  tokenization counter covers only a genuinely absent `TokenizedPrompt`. Configure `vllm`
  or `modelName` on the producer explicitly and verify the render sidecar answers before
  trusting a run.

- **When the prompt token count is absent, there is no ranking to fall back on.** This arm's
  profile is role-blind and carries no scorers, so no inherited pick exists. The arm returns
  the lowest-ID decode-eligible endpoint instead, which is a deterministic but arbitrary
  choice, and it is a third policy -- neither arm. Compare the tokenization counter between
  arms before reading any result: if the rates differ, the comparison is confounded.

- **`ablation.decomposed` alone is not the decode-first control from the literature.** It
  restricts the decode enumeration to whichever endpoint the inherited scores prefer, and
  with no scorers attached every score is 0, so the restriction lands on the lowest-ID
  decode pod -- a fixed arbitrary pod rather than a load-shaped choice. Reproducing the
  load-shaped decode-first arm additionally requires attaching load-shaped scorers to the
  profile, which is safe because the picker ignores `ScoredEndpoint.Score` when computing
  `J`. Flipping the switch on its own measures a weaker control that can resemble the
  intended one.

- **Resident state is reconstructed, not observed.** vLLM exports resident state only in
  aggregate, so the shadow table indexes the requests this EPP placed. Traffic bypassing
  this EPP is invisible to it, which under-counts the externality. The recorded first token
  is a dequeue instant, so realized TTFT is late, which shrinks a common positive
  multiplier on every charge. Both biases point toward local placement.

- **A TTL sweep is required, and it is approximate.** The `ResponseBodyProcessor` signature
  carries no error or termination state, so a client disconnect is indistinguishable from a
  completion. Without the sweep, entries from requests that never send a final chunk are
  charged as residents forever. With it, a legitimately long-lived request could be reaped
  early if the TTL is set below the workload's longest deadline.

- **`S_pf` is an over-estimate.** What is being prefilled in the current engine step is not
  observable from the EPP, so the shadow sum is capped per occupant and in total. The cap
  must equal `engine.chunkTokens`, and startup validation enforces that. The residual
  over-estimate over-states local prefill inflation.

- **The transfer price is a configured constant.** `xferBaseUs` and `xferBandwidthGbps` are
  not measured by this plugin, and the transfer price is the only size-dependent cost of
  going remote, entering the objective at exactly one place. A wrong value mis-prices
  systematically rather than noisily. Measure both before trusting any magnitude.

- **`FreeKVBlocks` is a floor.** It is derived as `(1 - KVCacheUsagePercent) * CacheNumBlocks`
  with integer truncation, on top of exporter quantization, so it understates free KV and
  over-states admission delay.

- **A zero SLO target flattens the policy rather than loosening it.** A disabled dimension
  contributes exactly one to the value, so a zero triple makes every resident charge
  `1.0 - 1.0 = 0` and the argmin degenerates to enumeration order while still reporting
  results. `tauE2eUs` is the sharper case: because a resident's realized TTFT is
  placement-invariant it factors out of the charge, so a zero E2E target makes the entire
  decode-side externality identically zero and reduces the arm to its own-good term.
  Startup validation rejects both.

- **The capacity term is implemented but disabled in the intended configuration.** Its
  algebra is present so the switch is reachable, but re-enabling it needs a monotonic clock
  and per-endpoint drain timestamps, and the commit path that feeds the admission
  estimator's waiting-work account is deliberately not maintained.

## Related Documentation

- [Disaggregation Architecture](../../../../../../../docs/disaggregation.md)
- [Disaggregated Profile Handler](../disagg/README.md)
