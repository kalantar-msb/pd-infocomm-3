# Configuration

The deployment this transfer targets, and every knob the arms read.

Three pins; every citation in this file and in `algorithms/` resolves against them:

| tree | repository | pin |
|---|---|---|
| simulation | `vishakha-ramani/inference-sim` | `871b169bb13934ca8dd1e002638e1f6bf490b3b5` (`infocom-implementation`) |
| target | `llm-d/llm-d-router` | `5f4e762f341a5196393ce79f8a57c3e1900c4a6b` (v0.9.0) |
| engine | `vllm-project/vllm` | v0.26.0 |

Simulator paths are relative to the simulation pin, target paths to the target pin.

> Citations to `generate_from_config.py` carry `lint-skip`: the consumer lives under a
> hidden `.claude/skills/` path, which `lint_citations.py` structurally excludes
> (`scripts/lint_citations.py:75` <!-- lint-skip -->), so no `--tree` can resolve them. Their authority is the
> Phase 6 consumer check, which runs the real generator against this file — not a line number.


---

## vLLM Pod Configuration

**This table is machine-read.** `/sim2real-bootstrap` Task 3 derives
`baselines/baseline.yaml` from it and halts without it. Do not add a second table
whose heading also matches `vllm pod configuration` — the parser takes the first
match (`generate_from_config.py:443-450`).  <!-- lint-skip -->

| Parameter | Value | Source |
|---|---|---|
| Model | `meta-llama/Llama-3.3-70B-Instruct` | `defaults.yaml:10-13` (`hf_repo` for the campaign's `--model`, `run_decisive_campaign.py:29`) |
| GPU | `H100_SXM_80GB` | fleet pool `fast`, `campaigns/edpp-study/inputs/hetero-realistic-1p2d.yaml:3` |
| prefill GPU | `H100_SXM_80GB` | placement, see §2 |
| decode GPU | `H100_SXM_80GB` | placement, see §2 — **this is one of two decode GPU types; the A100 decode instance is not representable here** |
| Number of prefill pods | 1 | `--prefill-instances 1`, `run_public_workload_heterogeneity_closeout.py:181-182,198-199` |
| Number of decode pods | 2 | `--decode-instances 2`, same sites |
| tensor_parallel_size | 4 | `defaults.yaml:12` (`tensor_parallelism: 4`) |
| max_num_seqs | 256 | `--max-num-running-reqs 256`, `run_public_workload_heterogeneity_closeout.py:200-201` |
| max_num_batched_tokens | 2048 | `--max-num-scheduled-tokens` default, `cmd/root.go:1426`; reaches the policy as `ChunkTokens` at `sim/cluster/cluster.go:525` |
| block_size | 16 | `--block-size-in-tokens` default, `cmd/root.go:1429` |
| max_model_len | 131072 | `model_configs/llama-3.3-70b-instruct/config.json` (`max_position_embeddings`) |
| gpu_memory_utilization | 0.9 | **operator-stated** — no campaign flag sets it; see §3.1 |
| enable_prefix_caching | true | required by `a_p`, §5 signal 11; all three workloads declare prefix groups |
| enable_chunked_prefill | true | the simulator chunks prefill against `MaxScheduledTokens` (`sim/batch_formation.go:125`) |
| enforce_eager | false | vLLM's own default; the campaign models no graph-capture penalty |

### Two lookup-table entries bootstrap needs before Task 3

Both are non-fatal warnings from `generate_from_config.py`, and both mean the
generated baseline is wrong rather than absent. Add them before trusting Task 3:

- **`model 'meta-llama/Llama-3.3-70B-Instruct' not in MODEL_METADATA`.** The model
  is absent from `generate_from_config.py:32-51` <!-- lint-skip -->, so `shortName`, `path`, `size`,
  and `maxModelLen` are derived rather than looked up. `max_model_len` is stated
  in the table above precisely because that pair — absent model *and* absent
  `max_model_len` — is what makes the omission fatal. Note `size`: `hf download`
  fetches the whole repo, so Llama-3.3-70B lands as ~132 GiB of safetensors plus
  ~131 GiB of `original/*.pth` that vLLM never reads. A 300Gi PVC is marginal and
  an existing PVC is never resized.
- **`hardware 'H100_SXM_80GB' not in HARDWARE_LABELS`** does *not* fire —
  `H100_SXM_80GB` and `A100_SXM_80GB` are both present
  (`generate_from_config.py:53-57`) <!-- lint-skip -->, mapping to `NVIDIA-H100-80GB-HBM3` and
  `NVIDIA-A100-SXM4-80GB`. Recorded because §2 needs both.

---

## Fleet topology

**The simulation fleet is not one homogeneous pool, and the shortfall is in the
decode role.** This section states what the campaign ran, what the scenario schema
can express, and what the difference costs.

### What the campaign ran

Two node pools (`campaigns/edpp-study/inputs/hetero-realistic-1p2d.yaml`):

| pool | GPU | GPUs/node | nodes |
|---|---|---|---|
| `fast` | H100 | 8 | 1 |
| `slow` | A100 | 4 | 1 |

Three instances at TP=4, roles assigned by index — `instance_0` prefill,
`instance_1..2` decode (`sim/cluster/pool.go:115-135`,
`BuildPoolMembershipFromIndices`). Placement is **first-fit over pools in
declared order** (`sim/cluster/infra_placement.go:184-228`), and the pool's
`gpu_type` overrides the CLI `--gpu` flag (`sim/cluster/cluster.go:384-395`). So:

| instance | role | lands on |
|---|---|---|
| `instance_0` | prefill | `fast` / **H100** (4 of 8 GPUs) |
| `instance_1` | decode | `fast` / **H100** (remaining 4) |
| `instance_2` | decode | `slow` / **A100** (4 of 4) |

**The decode pool is heterogeneous.** That is not incidental — it is the condition
the mechanism exploits. Per-GPU coefficients (§4) differ most in the
per-iteration intercept: 25 563.82 µs on A100 against 16 613.54 µs on H100, a
factor of 1.539 present on *every* iteration regardless of KV state.

### What the scenario schema can express

One `decode.acceleratorType.labelValue` and one `prefill.acceleratorType.labelValue`
(`generate_from_config.py:779-785, 819-833`) <!-- lint-skip -->. Per-role hardware is expressible;
**per-instance hardware within a role is not.** There is no schema construct for a
decode pool holding two GPU types, and `parallelism.workers` is a
LeaderWorkerSet group size, not a way to name a second variant
(`generate_from_config.py:344-348`).  <!-- lint-skip -->

So the machine-read table in §1 names `H100_SXM_80GB` as the decode GPU — one
value, so bootstrap parses and Task 3 produces a usable baseline — and that value
describes only `instance_1`. `instance_2`'s A100 physics is absent from the
generated scenario.

### What that costs, and the options

The heterogeneity is what gives the mechanism room: identical load signals on
non-identical hardware is exactly the case load-shaped scorers cannot see. A
homogeneous H100 decode pool is a *different experiment*, and the reproducibility
document's headline cells are on `h100_a100_realistic` — the worst single ablation
cell is `h100_a100_realistic:interactive:medium_0p80`.

| option | cost |
|---|---|
| **Two `InferencePool`s, one per decode GPU type** | Breaks the joint argmin outright. Two pools means two EPPs, each blind to the other's endpoints, so no argmin over the full decode set exists and the required shape (README, "Required structural shape") is gone. Rejected. |
| **One pool, per-pod `pd-infocomm.io/gpu-type` label; scenario names H100** | What §1 does. The *policy* reads hardware from the pod label, so it can price both GPU types correctly; only the *generated deployment* is single-variant. Requires the A100 decode pod to be deployed outside the generated scenario, or the scenario hand-edited after Task 3. This is the honest option and it is not fully automated. |
| **Homogenize to `h100_homogeneous`** | Expressible end-to-end, no hand editing. Loses the cells where the effect is largest; the transfer would measure the mechanism where it has least room. |
| **Extend the generator for per-instance decode hardware** | Removes the disclosure entirely. Blocks this bundle on a sim2real change. |

**Unresolved, and deliberately so:** this bundle does not choose among the last
three. §1 is written so bootstrap succeeds; the A100 decode instance is a stated
gap, not a silent one. The policy layer reads `Theta` from a pod label rather than
from the scenario (§5 signal 8), so a hand-added A100 decode pod is priced
correctly the moment it carries the label.

---

## 3. Simulation → deployment mapping

Each simulator flag beside the vLLM parameter it calibrates, so the two can be
audited against each other. The simulator dialect alone is not enough: a bundle
that records `--max-num-running-reqs 256` and stops has stated the fact and still
fails bootstrap, which reads `max_num_seqs`.

| simulator flag | value | vLLM parameter | agrees? |
|---|---|---|---|
| `--model` | `meta-llama/llama-3.3-70b-instruct` | `Model` (via `defaults.yaml` `hf_repo`) | yes |
| `--num-instances` | 3 | `prefill.replicas` + `decode.replicas` = 1 + 2 | yes |
| `--prefill-instances` | 1 | `prefill.replicas` | yes |
| `--decode-instances` | 2 | `decode.replicas` | yes |
| `--max-num-running-reqs` | 256 | `max_num_seqs` | yes |
| `--max-num-scheduled-tokens` | 2048 (default) | `max_num_batched_tokens` | yes |
| `--block-size-in-tokens` | 16 (default) | `block_size` | yes |
| `--policy-config` | `hetero-realistic-1p2d.yaml` | `acceleratorType` per role | **partially — see §2** |
| (`defaults.yaml` `tensor_parallelism`) | 4 | `tensor_parallel_size` | yes |

### 3.1 Deployment values the simulation cannot supply

No campaign flag sets these; they are properties of the cluster.

| parameter | value | note |
|---|---|---|
| `gpu_memory_utilization` | 0.9 | Operator-stated. Governs how much KV fits, so it bounds the deep-research workload's ~45k-token prompts at TP=4 on 80 GB cards. |
| `enable_prefix_caching` | true | Stated rather than left silent **on purpose**: bootstrap interprets silence as ON, so a bundle needing it OFF and staying quiet gets the opposite of what it meant. This bundle does need it ON — `a_p` (§5 signal 11) is the uncached suffix, and with no prefix residency every candidate is priced as cold. |
| `max_model_len` | 131072 | Stated because the model is absent from `MODEL_METADATA` (§1). |

---

## 4. Coefficients

Per-GPU-type `θ_i`, loaded by `LoadEDPPCoeffs` (`sim/edpp_coeffs.go:38-59`) and
selected per candidate by `coeffsFor(snap.GPUType)` (`sim/edpp.go:709`). Copied
verbatim into `inputs/`; the `source_csvs` paths inside them are provenance only.

| θ | symbol | H100 (`inputs/coeffs-llama70b-h100-tp4.json`) | A100 (`inputs/coeffs-llama70b-a100real-tp4.json`) | units |
|---|---|---|---|---|
| `AlphaD` | α | 16613.537554540144 | 25563.819286862163 | µs |
| `AlphaP` | α_p | 16617.85321583337 | 25568.34953836831 | µs |
| `C0` | c₀ | 5.347316038602452 | 5.945331876073271 | µs/req |
| `C1` | c₁ | 0.04761401141756073 | 0.07822809856114352 | µs/token |
| `CPf` | c_pf | 6.144687138665833 | 9.794219053944662 | µs/token |
| `CAttn` | c_attn | 0.00010075247918809842 | 0.00015977670754642687 | µs/token² |

`validate()` (`sim/edpp_coeffs.go:127-146`) rejects a pair whose α and α_p diverge
by more than 10%: both files are within 0.03%.

**On the target, θ is keyed by a pod label**, not by the scenario's
`acceleratorType` — see §5 signal 8 and §2. Both files must be transcribed into
every arm's plugin config, keyed by the label value, so the A100 decode instance
is priced correctly whether or not the generated scenario names it.

---

## 5. Signals the arms read

Where each quantity the decision rule reads comes from on the target.
**direct** = a native accessor returns it. **derived** = computed from native
accessors without loss. **degraded** = computed with a stated loss and a stated
direction of bias. **built** = no native signal exists; the port implements the
producer. **config** = no signal exists and none is derivable. **absent** = not
obtainable.

The target's entire native metrics surface is seven fields on one struct
(`pkg/epp/framework/interface/datalayer/metrics.go:26-42`). Everything below is
that, a pod label, config, or built.

| # | quantity | read at | status | route |
|---|---|---|---|---|
| 1 | `BatchSize` (B) | `sim/edpp.go:1690` | direct | `Metrics.RunningRequestsSize` (`pkg/epp/framework/interface/datalayer/metrics.go:32`) |
| 2 | `QueueDepth` | `sim/edpp.go:1607` | direct | `Metrics.WaitingQueueSize` (`pkg/epp/framework/interface/datalayer/metrics.go:33`) |
| 3 | KV block size | `sim/edpp.go:1257` | direct | `Metrics.CacheBlockSize` (`pkg/epp/framework/interface/datalayer/metrics.go:36`) |
| 4 | KV total blocks | — | direct | `Metrics.CacheNumBlocks` (`pkg/epp/framework/interface/datalayer/metrics.go:38`) |
| 5 | KV usage fraction | — | direct | `Metrics.KVCacheUsagePercent` (`pkg/epp/framework/interface/datalayer/metrics.go:34`) — a **fraction in [0,1]**, despite the name |
| 6 | `KvTokensInUse` (KV) | `sim/edpp.go:1690` | derived | `usage × blockSize × numBlocks` |
| 7 | `FreeKVBlocks` | `sim/edpp.go:1607` | degraded **D7** | `(1 − usage) × numBlocks`, truncated ⇒ a floor |
| 8 | `GPUType` → `θ` | `sim/edpp.go:709` | direct | pod label via `EndpointMetadata.Labels`; reject the endpoint on absent/unknown, **never default** |
| 9 | role (prefill/decode) | pool membership | direct | pod label, same accessor |
| 10 | `a_r` (prompt tokens) | `req.InputLen()`, e.g. `sim/edpp.go:1693` | built | tokenizer sidecar + token-producer plugin |
| 11 | `a_p` (uncached suffix, per endpoint) | `sim/edpp.go:1055-1064` | direct | prefix-cache producer, bound **by producer name** so both arms read one signal |
| 12 | per-resident decode state (`StepsDone`, `ArrivalUs`, `FirstTokenUs`, `TTFTSet`) | `sim/edpp_var.go:615-641` | **built**, degraded **D2** | EPP-side shadow table; `epp.replicas` pinned to 1 in the **baseline** so both arms match |
| 13 | pre-first-token occupants (`RunningPrefill`) | `sim/edpp_var.go:643-673` | **built**, degraded **D6** | same table, prefill index |
| 14 | `ResidentPrefillTokens` (S_pf) | `sim/edpp.go:1690` | degraded **D4** | capped shadow sum, total capped at `max_num_batched_tokens` |
| 15 | resident realized TTFT | `sim/edpp_var.go:632` | **built**, degraded **D2c** | response-body hook, first non-zero `Usage.CompletionTokens` |
| 16 | resident `StepsDone` | `sim/edpp_var.go:626` | **built** | `Usage.CompletionTokens` per chunk, monotonic — **requires `--enable-force-include-usage`**, else 0 forever |
| 17 | `N̂_out` per class | `sim/edpp.go:2343-2358` | **built** | per-class running mean over **completed** requests only, seeded at 1 |
| 18 | arrival timestamp | `sim/edpp_var.go:631` | **built** (biased) | routing instant, not client arrival |
| 19 | SLO class | `sim/edpp.go:682` | direct | request header |
| 20 | τ_ttft / τ_itl / τ_e2e | `sim/edpp.go:682-708` | config | §6; unresolvable ⇒ **startup failure**, never a zero triple |
| 21 | θ coefficients | `sim/edpp.go:709` | config | §4, keyed by pod-label value |
| 22 | `ChunkTokens` | `sim/edpp.go:1220-1228` | config | must equal `max_num_batched_tokens` = 2048 |
| 23 | `BlockSize` | `sim/edpp.go:1257` | config | must equal `block_size` = 16; **validate against scraped `CacheBlockSize` and fail loudly** |
| 24 | `MaxBatchSize` | `sim/edpp.go:1607` | config | no metric exists; must equal `max_num_seqs` = 256 |
| 25 | `OutputTokenProcessingTime` | `sim/edpp.go:1235-1242` | config | added to every TTFT projection; outside the calibrated θ |
| 26 | `c_xfer` | `sim/edpp.go:1126-1140` | **config, UNMEASURED** | §7 |
| 27 | `KVBytesPerTokenPerGPU` | `sim/edpp.go:1137` | derived (arithmetic) | 81 920 B — see §7 |
| 28 | scheduler queue contents | `sim/edpp_scheduler_rollout.go:79-243` | **absent D1** | no route; `rollforward` substitutes — see below |
| 29 | `PrefillTokensAhead` | `sim/edpp_kairos.go` | **absent K1** | needs a new vLLM gauge; **the Kairos arm is not registered** |

### D1 — the scheduler rollout, and the substitution

The published TTFT and admission clock replay the engine scheduler over its
ordered wait queue with per-request prompt and computed token counts
(`sim/edpp_scheduler_rollout.go:79-243`), reached from the focal arm at
`sim/edpp.go:1703`, `:1725`, and `:1735`. The target exposes
`Metrics.WaitingQueueSize` — **one integer** (`pkg/epp/framework/interface/datalayer/metrics.go:33`). There is no route
to the queue contents, the currently-scheduled grants, or the step start instant.

**Substitution:** `rollforward` (`sim/admission_estimator.go:126-176`), the
closed-form estimator, which is also what the simulation itself falls back to when
the rollout guard returns `ok == false`. Both estimators are stated in full in the
specification layer, so the substitution is legible rather than implied by an
absent branch.

**Direction of bias:** rollforward walks only the current running set's departures
and then falls back to a wave form (`sim/admission_estimator.go:167-172`). Past one
batch drain it *understates* admission delay ⇒ smaller `admissionSteps` ⇒
residents are charged for more of the arrival's interference ⇒ the local candidate
is **over**-priced ⇒ biased toward remote prefill.

**A substitution counter is required, not optional.** The fallback is acceptable;
the silence is the defect. Without a counter, running a different TTFT estimator
than the one that produced every number in `sim_results/` is invisible in the
goodput figure.

### D2 — the shadow table, and why replicas are pinned

The externality is a sum over *individual* residents' deadline slack
(`sim/edpp_var.go:615-641` reads per-resident `StepsDone`, `ArrivalUs`,
`FirstTokenUs`, `TTFTSet`). vLLM exports resident state only in aggregate. The
port therefore maintains an EPP-side index of the requests it placed.

`epp.replicas: 1` is pinned **in the baseline**, so both arms match. Two
degradations remain:

| | what | direction of bias |
|---|---|---|
| D2b | traffic bypassing this EPP is invisible — aggregate `RunningRequestsSize` still sees it, so batch size stays right while resident detail is short | under-counts the externality ⇒ biases **local** |
| D2c | the recorded first token is a *dequeue* instant, so realized TTFT is late | shrinks `u_ttft`, a common positive multiplier on both terms of a resident's charge ⇒ every charge shrinks ⇒ biases **local** |

D2c is worth the algebra rather than the intuition. Under the composite kernel a
resident's charge is `gDecodeComposite(cb) − gDecodeComposite(cp)`
(`sim/edpp_var.go:253-260`, `:319-338`), and because realized TTFT is fixed by the
placement decision it factors out as `σ((τ_ttft − ttft)/τ_ttft)` multiplying both
terms. A late first token shrinks that factor and scales the whole difference down.

**Net direction across all degradations is not known and is not claimed.** D1
biases toward remote; D2b and D2c bias toward local. Read this section as the list
of things to measure before trusting a magnitude — not as "the errors cancel".

---

## 6. SLO targets, per workload

Resolved per SLO class by `targetsFor` / `e2eFor` (`sim/edpp.go:682-708`) and
consumed as `varSLO` (`sim/edpp_var.go:597-605`). From
`run_public_workload_heterogeneity_closeout.py:60-105`:

| workload | τ_ttft | τ_itl | τ_e2e |
|---|---|---|---|
| `interactive` | 1 000 ms | 50 ms | 16 000 ms |
| `reasoning` | 2 000 ms | 100 ms | 802 000 ms |
| `deep_research` | 10 000 ms | 100 ms | 40 000 ms |

Three facts a port must not lose:

- **Every cohort in all three workloads declares `slo_class: standard`**
  (`workloads/interactive-chat-single-turn.yaml:21` and the other two). The class
  is one constant string across the whole grid, so a per-request lookup by SLO
  class cannot select the triple — the simulation varies τ *per invocation*.
  Carry all three triples in config and select one, then run one assemble per
  workload. Thresholds are then never retyped and cannot drift between arms.
- **An unresolvable selection must fail at startup.** A zero triple does not
  loosen the policy, it *flattens* it: `sloCompositeValue`
  (`sim/edpp_var.go:242-251`) returns 1.0 for every disabled target, so every
  resident charge becomes `1.0 − 1.0 = 0`, the externality is 0 on every
  candidate, and the argmin is a tie broken by enumeration order — while the
  policy keeps running and keeps reporting goodput.
- **`interactive` is the committed default.** The composite kernel reads only
  τ_ttft and τ_e2e (`sim/edpp_var.go:242-251`; `gCollocComposite` at `:560-571` is
  TTFT-only), and a resident's charge scales roughly as `1/τ_e2e`, so
  interactive's 16 s deadline gives ~50× the discrimination of reasoning's 802 s,
  where the E2E sigmoid is saturated and the externality stops separating
  candidates.

Reported goodput remains the **hard** conjunction of TTFT, mean ITL, and E2E.
Mean ITL is an evaluation gate, not a routing term — the composite routing kernel
never reads τ_itl.

---

## 7. Transfer model — MUST BE MEASURED

`c_xfer` is the **only** size-dependent price of going remote, entering at exactly
one place (`sim/edpp.go:1735`, `remoteLeadUs`). An unmeasured value therefore
mis-prices systematically rather than noisily.

| knob | simulator value | status |
|---|---|---|
| `XferBaseUs` | 50.0 | **unmeasured on the target** |
| `XferBandwidthGBps` | 25.0 | **unmeasured on the target** |
| `KVBytesPerTokenPerGPU` | 81920 | derived arithmetic, not measured |

`KVBytesPerTokenPerGPU` = 2 (K,V) × 80 layers × 8 KV heads × 128 head_dim × 2 B ÷
TP 4 = 81 920 B, from `model_configs/llama-3.3-70b-instruct/config.json`
(`num_hidden_layers`, `num_key_value_heads`, `head_dim`, `torch_dtype`) and TP=4.
Leaving it 0 is severe with `CXferSizeAware`: `transferBytes` becomes 0
(`sim/edpp.go:1138`) so every request is charged a flat `XferBaseUs`, and a
4k-token prompt should be charged ~13 500 µs — roughly 270× under-priced, with the
error growing linearly toward the deep-research workload's ~45k-token prompts.

**Measure both before trusting any magnitude**, and record the deviation as a
stated degradation. Under-pricing transfer biases the joint argmin toward remote
prefill.

---

## 8. Real-cluster load generator (`blis observe`)

Emitted even where a value equals the pipeline default, so the values are this
bundle's decision and not a downstream fallback.

```bash
blis observe \
  --max-concurrency 10000 \
  --timeout 1800 \
  --warmup-requests 50 \
  --prewarm-duration 60s
```

---

## 9. Per-arm settings

Three arms are specified. Two are registered; the third is specified and
deliberately not registered.

### 9.1 Shared by every arm

| knob | value | why |
|---|---|---|
| `Joint` | true | the required shape — one argmin over D local plus D×P disaggregated candidates (`sim/edpp.go:1380-1398`) |
| `ChunkTokens` | 2048 | = `max_num_batched_tokens`, §1 |
| `BlockSize` | 16 | = `block_size`, §1 |
| `MaxBatchSize` | 256 | = `max_num_seqs`, §1 |
| `TAdmEstimator` | `rollforward` | the D1 substitution, §5 |
| `CXferSizeAware` | true | §7 |
| θ per GPU type | §4 | keyed by pod label, §2 |

### 9.2 `causal_slo_externality` — focal

The campaign policy `causal_externality_no_capacity_v8`
(`run_public_workload_heterogeneity_closeout.py:109-112`).

| knob | value | why |
|---|---|---|
| `JointSLOExternality` | true | selects `jointSLOExternalityCandidateScore` (`sim/edpp.go:1688`) |
| `V` | 8 | `score.total = V·(externality − ownGood) + capacity` (`sim/edpp.go:1777`) |
| `SLOExternalityNoCapacity` | **true** | the `no_capacity` controller: capacity drift disabled, so `total = 8·(externality − ownGood)` |
| `SLOExternalityNoExternality` | false | keep the externality — it is the mechanism |
| `SLOExternalityNoOwnGood` | false | keep the arriving request's own good |
| kernel | `composite` | forced at `sim/edpp.go:1712` and `:1748` |
| `VarExactPrefillOverlap` | true (forced) | forced by the focal arm at `sim/edpp.go:1709` and `:1745`, overriding config |
| `varDeployable` | true (forced) | same sites — censored `N̂_out`, never true remaining |
| `varCollocPrefill` | true (forced) | same sites |

`V = 8` is **kept, not folded away**, even though with capacity disabled it is a
common positive multiplier that cannot change the argmin: the ablation cohort's
validity gate asserts `score = 8 × (externality − own_good)` exactly, and a port
that folds `V` away cannot reproduce that check.

### 9.3 `least_ttft_joint` — comparator

`--edpp-rule least-ttft` with `--edpp-ttft-overlap-aware`
(`run_decisive_campaign.py:325-331`). Shares the candidate set, the estimators,
the physics, and the prefix reads; differs **in the objective only**. That is what
attributes the effect to the mechanism rather than to the machinery.

Its routing view must be a **verbatim copy** of the focal arm's, package clause
aside. A re-derived-but-slightly-different estimator would silently destroy the
attribution argument while every test still passed.

### 9.4 `kairos_paper` — external comparator, NOT registered

`--edpp-rule kairos-paper`, α = 1.3, **β = 1.0** — hardcoded together at
`run_decisive_campaign.py:398-403`. The campaign policy is
`kairos_paper_alpha_1p3` (`run_public_load_static_benchmark.py:42`), the arm that
appears in the README's minimax ranking.

**Not `kairos_beta_0p5`.** That policy emits `--edpp-rule kairos --kairos-beta 0.5`
(`run_decisive_campaign.py:396-397`) — the *adapted* alias, a different algorithm.
The two are mutually exclusive: paper mode only ever ships with β = 1.0. Deploying
β = 0.5 into a paper-mode plugin halves the TBT budget and silently suppresses
deflection.

**Not registered**, because `PrefillTokensAhead` (§5 signal 29) has no route and
reading zero does not add noise — it *inverts* the comparator. Both effects push
the same way, so a zero-filled run measures a more aggressive relative of Kairos in
the direction that makes it look worse: a misreport, not a partial result. It is
specified so the algebra is on record and so the arm can be added with
`sim2real translation append` once a `num_pending_prefill_tokens` gauge exists.

---

## 10. Ablation switches

Cited evidence, from `sim_results/ablation/`. These are the comparisons that
isolate the mechanism; each is a config switch, so reproducing one on the target
needs no code change.

Read the sign as **how much the full policy gains from having the term**, which is
the direction
`campaigns/edpp-study/results/infocom-2027/ablation/PUBLIC-EXTERNALITY-DECOMPOSITION-ABLATION.md:19-21`
reports (copied verbatim to `sim_results/ablation/`).

| ablation arm | knob that produces it | term it removes | full policy's gain from that term | 95% CI |
|---|---|---|---:|---|
| `joint own-only` | `SLOExternalityNoExternality` | the resident externality | **+0.0154** | [+0.0112, +0.0195] |
| `joint resident-only` | `SLOExternalityNoOwnGood` | the arriving-request own good | **−0.0016** | [−0.0053, +0.0022] — **crosses zero** |
| `decode-first full` | `DecomposedSLOExternality` | the joint shape | **+0.0485** | [+0.0305, +0.0665] |

Two things follow, and neither is comfortable:

- **Only the resident externality is supported.** Removing it costs 0.0154, and the
  interval excludes zero. That term is the mechanism.
- **The own-good term is not supported.** Removing it *gains* 0.0016 and the interval
  crosses zero, so `joint resident-only` reaches 0.9280 equal-cell mean against the
  full policy's 0.9264 (and 0.7844 worst cell against 0.7688). The focal arm keeps
  the term anyway (§9.2, `SLOExternalityNoOwnGood: false`), so the transfer carries
  it — see the README's honest-status section.

The decode-first row is the cheapest on-cluster check that the joint shape
survived the port: flipping it should cost what the simulation says it costs.

---

## 11. Runtime requirements on the target

Settings without which a signal is silently wrong rather than absent.

| setting | signal | silent failure if omitted |
|---|---|---|
| `--enable-force-include-usage` | 15, 16 | `Usage.CompletionTokens` arrives only in the final chunk ⇒ `StepsDone` stays 0 for every request's whole lifetime ⇒ every remaining-steps estimate wrong |
| tokenizer sidecar reachable | 10 | `a_r` returns **0 with no error** ⇒ every prompt reads as shorter than any threshold ⇒ no disaggregation ever chosen |
| `epp.replicas: 1`, in the **baseline** | 12–18 | replicas partition the shadow table (D2) |
| one `InferencePool` for all decode endpoints | 8, and the argmin itself | two pools ⇒ two EPPs, each blind to the other's endpoints ⇒ no joint argmin exists |
| pod label carrying GPU type, `nodeSelector` pinned to the same value | 8 | a mismatch serves traffic under the wrong physics; pinning makes it fail at *scheduling* time instead |
| `enable_prefix_caching: true` | 11 | no prefix residency to report ⇒ every candidate priced as cold |
| block size validated against scraped `CacheBlockSize` | 23 | a config/engine disagreement leaves a latent unit bug in the admission test, which compares blocks accumulated in one unit against a need expressed in another |

Two traps confirmed against the target pin, both silent:

- **`Metrics.KVCacheUsagePercent` is a fraction in `[0,1]`, not a percent.**
  Dividing by 100 under-counts KV by 100× and collapses the `C1·KV` term that
  carries the hardware heterogeneity this experiment exists to measure.
- **`Metrics.KvCacheMaxTokenCapacity` is unusable.** Declared at
  `pkg/epp/framework/interface/datalayer/metrics.go:35` and copied in `Clone` at
  `:77`, it is assigned nowhere else outside tests — so it always reads 0 and any
  branch guarded on it being positive is dead code. Derive KV capacity from
  `CacheNumBlocks × CacheBlockSize` instead (signal 6).
