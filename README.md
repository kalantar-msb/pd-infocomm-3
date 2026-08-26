# pd-infocomm-3 — causal SLO externality, sim2real transfer bundle

A specification bundle for transferring the INFOCOM 2027 joint
causal-SLO-externality routing policy from the BLIS discrete-event simulator to
llm-d-router.

Upstream provenance: `INFOCOM_REPRODUCIBILITY.md` on the `infocom-implementation`
branch of `vishakha-ramani/inference-sim`.

This bundle is the **input** to `/sim2real-bootstrap`. It contains no
`transfer.yaml`, no `baselines/`, and no compiling code — `algorithms/` is a
specification layer by design, and its target-API adapters are declared unknowns
so a wrong guess fails loudly rather than mis-scoring silently.

## Pins

| tree | repository | pin |
|---|---|---|
| simulation | `vishakha-ramani/inference-sim` | `871b169bb13934ca8dd1e002638e1f6bf490b3b5` (`infocom-implementation`) |
| target | `llm-d/llm-d-router` | `5f4e762f341a5196393ce79f8a57c3e1900c4a6b` (v0.9.0) |
| engine | `vllm-project/vllm` | v0.26.0 |

The simulation worktree was clean at that commit. All 16 files in `sim_results/`
verify against `sim_results/CHECKSUMS.sha256`, with digests identical to the
upstream manifest at that pin; the manifest was rewritten with bundle-relative
paths and excludes upstream's `README.md`, which is provenance rather than a
gated result.

## The mechanism

A joint argmin over (decode endpoint, prefill placement) of

```
J(d, p) = V * ( externality(d, p) - ownGood(d, p) ),    V = 8
```

with the capacity term disabled (the `causal_externality_no_capacity_v8`
controller).

**What differs between conditions** is the SLO value already banked in a
candidate endpoint's residents. Two endpoints with identical queue depth and
identical KV occupancy are not interchangeable: one may hold residents sitting
just inside their end-to-end deadline, where one extra co-scheduled prefill flips
them to a miss; the other may hold residents with comfortable slack, or residents
already doomed, where the same prefill costs nothing recoverable. Load-shaped
signals — queue depth, KV utilization, projected TTFT — cannot see that
difference, because it is a property of the residents' deadlines rather than of
the endpoint's load.

**The mechanism that exploits it** prices each candidate by the smooth SLO value
it destroys among residents, evaluated *causally*: every completion model is gated
on an admission window, so a resident that finishes before the arrival is admitted
contributes exactly zero. That gating is what makes the term an externality rather
than another load proxy, and it is why the effect survives on a fleet whose load
signals are already balanced. Heterogeneous fleets give it the most room, because
per-GPU coefficients make the same resident set cost different amounts on
different hardware.

The routing value is a smooth TTFT × E2E surrogate. Reported goodput remains the
**hard** conjunction of TTFT, mean ITL, and E2E — mean ITL is an evaluation gate,
not a routing term, and the composite kernel never reads τ_itl.

## Required structural shape

A **joint argmin over the cross product**: D local candidates plus D×P
disaggregated candidates, scored on one scale and compared in one argmin.

This cannot be expressed as a per-endpoint scorer on llm-d-router v0.9.0, for two
independent reasons that must be confirmed against the pinned checkout:

1. a scorer returns a per-endpoint map and has **no way to name a pair**;
2. every scorer contribution passes through the profile's score-range
   enforcement, which **clamps to [0,1]**, while `J` is signed and unbounded.
   Clamping does not degrade the ranking, it destroys it — every candidate with
   `J ≤ 0` collapses to one score.

So the policy must attach as a **custom picker over a role-blind profile**, and it
must ignore inherited scorer contributions for the same clamping reason. The
comparator faces the same hazard for a different reason: its objective is a
latency in microseconds, non-negative and unbounded above, so clamping to [0,1]
would collapse essentially every candidate too.

**The cost of getting this wrong is measured, not assumed.** The target's natural
decomposition is decode-first: pick a decode endpoint with the stock scorer, then
decide placement. `sim_results/ablation/` prices that at **+0.0485** equal-cell
mean goodput in favour of the joint shape, 95% CI [+0.0305, +0.0665], with the
worst single cell falling `0.918 → 0.700` at
`h100_a100_realistic:interactive:medium_0p80`. The hazard is a **silent fallback**
to the weaker decomposition, which transfers a different algorithm while
resembling success — so the decomposition switch is kept reachable
(`config.md §10`), and flipping it is the cheapest on-cluster check that the joint
shape survived the port.

## Honest status

### What the simulation shows

Deployable minimax ranking, `sim_results/main/`. The claim is a **worst-regret**
claim against frozen static plans, and the focal arm also happens to lead on mean:

| rank | policy | worst regret | worst cell | equal-cell mean goodput |
|---:|---|---:|---|---:|
| 1 | `causal_externality_no_capacity_v8` | 0.0100 | `h100_homogeneous:interactive:medium_0p80` | 0.9209 |
| 2 | `least_ttft_joint` | 0.0542 | `h100_homogeneous:interactive:high_0p95` | 0.9022 |
| 3 | `kairos_paper_alpha_1p3` | 0.1050 | `h100_a100_realistic:interactive:high_0p95` | 0.9030 |

The stress cohort (`sim_results/stress/`) gives 0.0031 against 0.1110 (least-TTFT)
and 0.0924 (Kairos); the topology sweep (`sim_results/topology/`) 0.0100 against
0.0800.

### Where the policy did NOT win

This matters more than the headline, because the port carries these terms anyway.

- **The arriving-request own-good term is not supported.** Adding it is worth
  **−0.0016**, 95% CI **[−0.0053, +0.0022]** — the interval crosses zero
  (`sim_results/ablation/`). The focal arm nonetheless **keeps** it
  (`SLOExternalityNoOwnGood: false`), so the transfer carries a term with no
  measured benefit. Only the *resident* externality has a supported effect:
  **+0.0154**, CI [+0.0112, +0.0195].
- **The resident-only ablation beats the full policy on aggregate**: 0.9280
  equal-cell mean and 0.7844 worst cell, against the full policy's 0.9264 and
  0.7688. It gets there by deflecting far more (mean remote fraction 0.8944
  against 0.4328).
- **On the homogeneous fleet the full policy is at or slightly below
  resident-only** in most cells, and its worst regret cell
  (`h100_homogeneous:interactive:medium_0p80`) is on that fleet — where the
  `llm-d threshold` baseline scores 0.963 against the focal arm's 0.953, and the
  frozen goodput-static plan reaches 0.991 against 0.980 at `low_0p60`.
- **On reasoning cells the arms are indistinguishable.** At
  `h100_a100_realistic:reasoning:low_0p60` the focal arm scores 0.991 against
  0.994 for both comparators. τ_e2e = 802 s saturates the E2E sigmoid, so the
  externality stops separating candidates — which is why `interactive` is the
  committed default (`config.md §6`).

The mechanism's room is on the **heterogeneous** fleet at **interactive**
deadlines. That is exactly the configuration the target's scenario schema cannot
fully express — see below.

### Pre-registered expectation

Stated before any target measurement, so the transfer can be wrong.

1. **Ordering, not magnitude.** The focal arm should show a smaller worst-cell
   regret than `least_ttft_joint` on the heterogeneous fleet at interactive
   deadlines. No margin is pre-registered: the declared degradations do not point
   the same way, so the simulation's 0.0100-vs-0.0542 is not a prediction about
   the cluster.
2. **The joint shape should cost what the ablation says.** Flipping the
   decomposition switch should lose goodput of roughly the ablation's order. If it
   costs nothing, the joint argmin is probably not actually running.
3. **Reasoning cells should show no separation.** If they do, something other than
   the mechanism is moving.
4. **Kairos is not measured.** Its arm is specified and deliberately unregistered
   (K1, below), so no Kairos number should appear in target results at all.

### Open threats to validity

Named, with direction where known. **Net direction is not known and is not
claimed** — these do not point the same way, and this list is what to measure
before trusting any magnitude, not an argument that the errors cancel.

| | threat | direction |
|---|---|---|
| **D1** | The scheduler rollout that produced every published number is unobtainable; `rollforward` substitutes | **arm-dependent, not one sign.** In the focal arm, understating admission delay ⇒ toward **remote**. In the comparator, whose disaggregated branch joins clocks with a `max` while local adds with slope 1, understating *decode* admission biases **local** or cancels, and only the *prefill* channel biases remote |
| **D2b** | Traffic bypassing this EPP is invisible to the shadow table | under-counts externality ⇒ toward **local** |
| **D2c** | Recorded first token is a dequeue instant, so realized TTFT is late | shrinks a common positive multiplier on every charge ⇒ toward **local** |
| **D4** | `ResidentPrefillTokens` is an EPP-side estimate, still an over-estimate | over-states local prefill inflation ⇒ toward **remote** |
| **D5** | `c_xfer` is **unmeasured** — the only size-dependent remote price | under-pricing ⇒ toward **remote** |
| **D6** | Pre-first-token occupants incomplete | under-counts ⇒ toward **local** |
| **D7** | `FreeKVBlocks` is a floor | over-states admission delay |
| **D8** | Tokenization can be unavailable at runtime; on failure an arm makes no decision at all | not a routing bias — a silent fallback to the stock scorer, i.e. a **third policy**. Confounds the comparison if the rate differs between arms |
| **Fleet** | The decode pool is genuinely heterogeneous and the scenario schema cannot represent that | **unresolved — see `config.md §2`** |
| **Own-good** | Carried despite a CI crossing zero | unknown; it is a term with no measured benefit |

Each degradation is declared in the specification layer's own header with its
direction of bias. The fleet item is not a bias — it is a **capability gap** in
the deployment path, disclosed with options and costs rather than silently
homogenized.

## Layout

| path | what |
|---|---|
| `algorithms/causal_slo_externality.go` | focal arm — the complete objective, kernels, estimators, and rollout |
| `algorithms/least_ttft_joint.go` | comparator — same machinery, objective only differs; carries a verbatim-copy contract |
| `algorithms/kairos_paper.go` | external comparator, **specified but must not be registered** (K1) |
| `config.md` | the deployment, every knob, the signal inventory, and the fleet-topology disclosure |
| `workloads/` | the three closeout workloads, copied verbatim |
| `inputs/` | fleet definition and both per-GPU coefficient files, copied verbatim |
| `sim_results/` | the 16 tracked result artifacts plus `CHECKSUMS.sha256` |

### Why Kairos is specified but not registered

`PrefillTokensAhead` — the prefill pool's total remaining prompt tokens — has no
route at this pin, and reading zero does not add noise, it **inverts** the
comparator. The eligibility gate is a disjunction, so zeroing the second operand
degrades it from "exclude any node with a deflected prefill in flight" to "exclude
only a node being chunk-prefilled this step"; and the pool queue-wait term
collapses to zero, making remote prefill look free and making the margin test
harder to pass. Both push the same way, so a zero-filled run measures a more
aggressive relative of Kairos in the direction that makes it look worse. That is a
**misreport, not a partial result**.

Unblocking it needs no llm-d-router change: a new vLLM gauge summing the wait
queue's remaining prompt tokens would ride the existing metrics path into endpoint
attributes. Add the arm at that point.

## Next

```
/sim2real-bootstrap <this directory>
```
