# least-ttft-joint — the comparator arm

The least-TTFT joint comparator for the INFOCOM 2027 transfer (`config.md` §9.3). Its
counterpart is the focal arm in `../causalsloexternality`.

## Why this arm exists

It shares the candidate set, the estimators, the physics, and the prefix reads with the
focal arm, and differs **in the objective only**. That is what attributes the measured
effect to the *mechanism* rather than to the *machinery*: a weaker baseline would leave
open that the focal arm won because it computes better physics, not because it prices SLO
externality.

Cited evidence for the comparison (`INFOCOM_REPRODUCIBILITY.md` expected checkpoints,
reproduced in `sim_results/main/`), worst static-plan gap:

| arm | main cohort | stress cohort |
|---|---:|---:|
| causal externality | 0.0100 | 0.0031 |
| least-TTFT | 0.0542 | 0.1110 |

## The objective

The argmin of the arriving request's **own** projected time-to-first-token over the same
candidate set — `Policy.jointCandidateTTFT`. Given a decode endpoint `d` and an optional
prefill placement `p`:

```
local   J = tAdmD + nChunks*tIterD + Wp + postProcessing
disagg  J = max(tAdmP + nChunksP*tIterP + WpP + cXfer,  tAdmD)
            + tIterFirstDecode + postProcessing
```

It carries no backlog drift, no SLO virtual queues, **no externality over residents**,
**no tau at all**, and no transfer penalty beyond the transfer latency already inside the
disaggregated projection.

Two details are load-bearing and easy to port wrong:

- **`max`, not `+`.** Remote preparation and decode-queue drainage are concurrent clocks
  from the routing instant. Serializing them over-prices every disaggregated candidate by
  up to a full decode admission delay. The `max` is **unconditional** — the campaign's
  `--edpp-ttft-overlap-aware` flag gates only upstream's *reduced* path and is a no-op
  here, so no config knob reaches it.
- **The local projection omits `tIterFirstDecode`.** Local execution samples its first
  token when prefill completes, so no decode iteration precedes it. That term's KV
  component scales with the request's own input length, so prompt size prices the remote
  path through *two* channels, not one.

Each side prices with its **own** candidate's theta, so the arm is hardware-aware by
construction. That is deliberate: a hardware-blind least-TTFT baseline would lose to the
focal arm partly because it cannot see the fleet, confounding the mechanism with hardware
awareness.

## The verbatim-copy contract, enforced

`config.md` §9.3 requires this arm's routing view to be a verbatim copy of the focal arm's,
package clause aside. This is not style. A re-derived-but-slightly-different estimator
would silently destroy the attribution argument **while every behavioural test still
passed** — the arms would then differ in the objective *and* in the physics, and no
measurement could separate them.

`contract_test.go` enforces it mechanically:

| files | property |
|---|---|
| `coeffs.go`, `admission.go`, `rollout.go`, `shadow.go` | **byte-identical** to the focal arm's, after rewriting the package clause |
| `types.go`, `shared.go` | **deletion-only** subsets of the focal arm's `types.go` and `policy.go` — every retained line appears in the focal file, in order |

Editing a shared symbol in *either* arm without editing the other fails
`go test ./pkg/epp/framework/plugins/scheduling/profilehandler/leastttftjoint/...`.

The same file also checks that the two arms cannot collide: distinct plugin type strings,
distinct decision-attribute key, distinct metric families, and — the one that would
otherwise fail EPP startup — that **both arms' handlers register into one Prometheus
registry** in either order.

Because byte-identity is what the test can check, the four exactly-copied files retain the
focal arm's authorial voice. In them, "this arm" means the *focal* arm, and their comments
refer to the externality, the value kernels, and the capacity account — none of which exist
here. That is deliberate; see `doc.go`.

## What is deliberately absent

This arm drops fields and files rather than carrying unread ones, because a
populated-but-never-read field is a reader trap: it implies a consumer exists somewhere.

- No `kernels.go` — no value kernels (`sloCompositeValue`, `gDecodeComposite`, `goodSelf`).
- No `capacity.go` — no capacity account.
- No `retiming.go` — nothing here needs `reTiming`.
- `Config` drops the tau triple, `V`, `Ablation`, and `Capacity`. Each absence is
  *unreachable* rather than merely unused; see `Config`'s doc. In particular the §10
  decomposition ablation is run on the **focal** arm, which is where it belongs — it prices
  the joint *shape*, and both arms share the shape.

The plugin decoder is strict (`DisallowUnknownFields`), so pasting the focal arm's
parameters block here is a startup failure rather than a silent partial apply. An operator
must not be able to set a tau triple on this arm and believe it took effect.

## Degradations inherited

The arm reads no resident *value* but does read the resident *populations*, so the shadow
table and its degradations come with it.

| id | where it enters | direction |
|---|---|---|
| D1 | every admission estimate (the rollout is unreachable at this pin) | **not a single sign** — see below |
| D2 | `RunningDecode` → `decodeRemStepsEst` and the rollforward KV walk | over-states delay |
| D4 | `ResidentPrefillTokens` → `S_pf` on every candidate | biases remote |
| D5 | `cXfer`, at exactly one place | systematic, unmeasured |
| D6 | `RunningPrefill` → `prefillRemStepsEst` | over-states prefill delay |
| D7 | `FreeKVBlocks` is a floor | over-states delay |
| D8 | nil `TokenizedPrompt` → no ranking | falls to a *third* policy |

**D2c is not inherited.** No kernel here reads `ArrivalUs`, `FirstTokenUs`, or `TTFTSet`.
Those fields are still carried by the verbatim-copied shadow machinery and read by nothing
in this package — `TestObjectiveIgnoresResidentValueFields` pins that.

**D1's direction is not a single sign for this objective**, and the focal arm's one-line
summary must not be copied over. Local *adds* `tAdmD` with slope 1 while disaggregated takes
`max(remoteLead, tAdmD)`:

- when `tAdmD` dominates the max, understating it is charged identically to both placements
  and **cancels**;
- when `remoteLead` dominates, disagg is insensitive to `tAdmD` altogether, so understating
  it lowers **only** the local candidate — biasing toward **local**, the opposite of the
  focal arm's direction.

The toward-remote direction survives only through the `tAdmP` channel, and only while
`remoteLead` dominates. This bears on the *attribution argument*, not just on this arm's
accuracy: it would be convenient to say both arms inherit D1 identically so the comparison
stays fair, but a load-dependent direction cannot support that claim. Under decode-side
congestion — the regime the high-load and burst cohorts target — D1 shifts the two arms'
local/remote splits by different amounts.

**Treat the comparison as fair only where the substitution counters show the same estimator
regime on both arms.** And the rollout was the *live* path upstream for this arm, not a
fallback, so on the target both arms run an estimator that produced none of the numbers in
`sim_results/`.

**D8 is sharper for the comparison than for either arm alone.** On a nil prompt this arm
returns no ranking, so a third policy — neither arm — decides. The arms are separate plugin
instances with separate producer bindings, so a difference in *decline rate* confounds the
comparison itself. It is likeliest on long prompts, exactly where local-versus-remote is
most contested.

## Observability

Ten metric families, all `least_ttft_joint_*`, all labelled `plugin_name` / `plugin_type`.
The two that gate reading any result at all are
`least_ttft_joint_estimator_substitution_total` (D1) and
`least_ttft_joint_tokenization_unavailable_total` (D8).

The families are namespaced to this arm rather than shared with the focal arm because the
focal arm's collectors are package-private and its `registerMetrics` tolerates only a repeat
of its own collector object — same-named collectors in two packages would fail registration
and therefore fail EPP startup. The cost is that the cross-arm D1/D8 comparisons are two
queries rather than one label selector; the metric names, label sets, semantics and
histogram buckets are deliberately identical so the two families line up term for term.

## Placement

Two plugin registrations sharing one state object, not one plugin with two interfaces:
`ProfileHandler.Pick` and `Picker.Pick` share a method name with different signatures, so no
single Go type can satisfy both.

```
least-ttft-joint-handler   ProfileHandler + PreRequest + ResponseBodyProcessor   (owns the state)
least-ttft-joint-picker    Filter + Picker                                       (resolves the handler by name)
```

The handler must be declared **before** the picker: by-name references resolve backward
only, and the picker's factory errors on nil rather than constructing a second, divergent
shadow table.

`Filter` is required, not incidental — `Picker.Pick` receives no request, so the arriving
request's token count reaches the argmin no other way. The argmin is a **Picker** and not a
Scorer for two independent reasons: a scorer returns a per-endpoint map and cannot name a
*pair*, and every scorer contribution is clamped to `[0,1]` while this objective is a
latency in microseconds, unbounded above.

Only **one** `ProfileHandler` is permitted per EPP, and the focal arm declares one too, so
naming both arms in a single plugin config is a startup error rather than a silent A/B. The
two arms run as two scenarios in two processes and never share a shadow table, a `Policy`, or
a decision — which is also why the two overlays are the only place their shared-value
agreement lives. `TestTheTwoOverlaysAgreeOnEverySharedKey` reads both shipped artifacts and
checks it.
