// Package randomdisagg specifies the fixed-probability random P/D control arm for
// the INFOCOM 2027 transfer.
//
// THIS FILE IS A SPECIFICATION LAYER, NOT COMPILING CODE. Bodiless functions are
// declarations of intent; the compiling implementation is the `random-pd-decider`
// plugin, whose source is carried at plugins/randomdisagg/ in this bundle.
//
// # PINS
//
//	simulation  NONE -- SEE R1. THIS ARM HAS NO SIMULATED COUNTERPART.
//	target      llm-d/llm-d-router  71f4f0999f95b96c49a9d0c4afbd18dfdb943c26  (v0.10.0)
//	engine      vllm-project/vllm   v0.26.0
//
// # THIS ARM IS A CONTROL, NOT A TRANSFER
//
// Every other arm in this bundle is a port: an algorithm the simulation discovered,
// carried onto the target so the two numbers can be compared. This arm is the
// opposite. It was never simulated, it discovers nothing, and it is not a candidate
// for deployment. It exists to be beaten.
//
// # WHY THIS ARM EXISTS
//
// TWO REASONS, AND THE SECOND IS THE IMPORTANT ONE.
//
// FIRST -- THE BASELINE'S THRESHOLD IS UNTUNED, so baseline-vs-anything is a
// comparison against an arbitrary operating point. The baseline runs
// `prefix-based-pd-decider` at `nonCachedTokens: 512`, which is llm-d's own
// documentation value, NOT the value the simulation arm
// `llmd_prefix_threshold_workload_tuned` was tuned to -- that arm tuned PER
// WORKLOAD and no file in sim_results/, config.md, or README.md records the tuned
// numbers. The baseline overlay says so itself and instructs the reader to treat
// baseline-vs-sim comparisons as unanchored. This arm has NO threshold. Its only
// knob is a rate, so it cannot be mis-tuned -- only mis-swept.
//
// SECOND -- IT SEPARATES *WHICH* REQUESTS FROM *HOW MANY*. A P/D policy does two
// things at once: it picks a disaggregation RATE, and it picks WHICH requests get
// disaggregated. Those two effects are confounded in every arm-vs-baseline
// comparison in this bundle, because the arms differ in both at once. Running this
// arm at the focal arm's OWN realized rate holds the first fixed and isolates the
// second:
//
//	if focal(p_realized) ~= random(p_realized), the selection logic contributes
//	nothing and the entire measured effect is the disaggregation rate.
//
// That is the cheapest strong falsification test available for the focal arm's
// mechanism, and nothing else in the bundle performs it. In the one cell of run
// `try2` that completed, the focal arm disaggregated 41 of 546 handled requests,
// so p = 0.075 is the matched-rate operating point for interactive-chat. MEASURE
// THE RATE PER WORKLOAD RATHER THAN REUSING 0.075 -- the focal arm's realized rate
// is a function of the prompt-length distribution, and the three workloads differ.
//
// # THE DECISION
//
// The whole algorithm, for probability p and request r:
//
//	disaggregate(r) = U(r) < p        U(r) in [0,1)
//
// p = 0 is exactly the simulator's `never` rule and p = 1 is exactly its `always`
// rule (inference-sim sim/disaggregation.go:70-74). This arm is the interpolation
// between two rules the simulator already has, which is the sense in which it is a
// control rather than a new idea.
//
// U IS DERIVED BY HASH, NOT DRAWN FROM A STREAM, and this is a correctness
// requirement rather than a style choice. Three reasons, in increasing severity:
//
//  1. REPRODUCIBILITY UNDER CONCURRENCY. A sequential RNG's position depends on
//     the order requests reach the plugin. The EPP serves concurrently, so that
//     order is nondeterministic and a stream-drawn sequence is not replayable even
//     at a fixed seed.
//  2. NO LOCK. A shared stream needs a mutex on the decision path.
//  3. STABILITY UNDER RE-ENTRY. The verdict must be a property of the request, not
//     of when it was asked. Hashing makes repeated calls for one request agree by
//     construction; a stream would silently answer differently each time.
//
// # WHAT THIS ARM READS: NOTHING
//
// Not queue depth, not KV utilization, not the prefix-cache hit fraction, not the
// GPU-type label, not the per-GPU coefficients, not tau, not the endpoint it is
// handed. The `endpoint` argument is accepted and ignored.
//
// THAT IS THE DEFINITION OF THE ARM, NOT AN OVERSIGHT. A control that reads a
// signal is not a control. The consequence is stated as R4 below: it cannot back
// off, so at saturation it is expected to be actively worse than the baseline.
// A run in which this arm wins at high load is evidence about the target's
// scheduler, not about this arm.
//
// # DECLARED DEGRADATIONS
//
// R1 -- NO SIMULATED COUNTERPART, SO NO SIM-VS-REAL PARITY FOR THIS ARM.
// The other arms cite sim/edpp.go at vishakha-ramani/inference-sim
// 871b169bb13934ca8dd1e002638e1f6bf490b3b5 (infocom-implementation) and have a
// sim_results/ number to be checked against. This arm has neither. It is
// REAL-ONLY BY CONSTRUCTION. Its comparisons are arm-vs-arm on the target and are
// valid there; any statement of the form "the simulation predicted this arm would
// ..." is unsupportable. Adding a `random` case to the simulator's rule dispatch
// would be a small change, but it would land in a DIFFERENT fork from the pinned
// one and still would not produce a number comparable to the campaign's
// sim_results/, so it is deliberately not claimed here.
//
// R2 -- EXACT REPLAY DEPENDS ON A STABLE REQUEST IDENTITY, WHICH IS NOT GUARANTEED.
// The implementation keys U on `IdentityHeader` when that header is configured and
// present, and otherwise on scheduling.InferenceRequest.RequestID, which
// llm-d-router documents as "the Envoy generated Id for the request being
// processed" -- freshly generated per request, per run. Under the fallback the arm
// is still correctly Bernoulli at rate p and every comparison remains valid, but
// two runs of the same workload will disaggregate DIFFERENT requests. Set
// IdentityHeader to a load-generator-stable header to recover bit-identical
// replay, and record which path was used when reporting.
//
// R3 -- THIS ARM NEEDS REPLICAS MORE THAN THE DETERMINISTIC ONES DO.
// Its realized rate is Binomial(n, p), so the rate itself carries sampling error
// of sqrt(p(1-p)/n) -- at n = 250 and p = 0.075 roughly +/-1.7pp -- on top of the
// usual run-to-run variance. A single iteration per cell cannot separate this
// arm's mean from noise. Runs assembled with `--replicas 1` are not adequate for
// it, and a matched-rate claim in particular should not be made from one cell.
//
// R4 -- IT CANNOT RESPOND TO LOAD. See "WHAT THIS ARM READS". Blind disaggregation
// at high saturation can push work toward an already-busy prefill pool. Expected,
// and part of what the sweep measures; not a defect to be fixed by adding a signal
// (that would make it a different arm).
//
// R5 -- THE FRAMEWORK WILL NOT RECORD THIS FILE.
// This arm enters the pdinfocomm3 translation through `sim2real translation
// append`, which writes appended entries with source_path = null and
// source_sha256 = null (pipeline/sim2real.py, _append_translation). So unlike the
// other two arms, NOTHING IN translation_output.json LINKS THE RUNNING PLUGIN TO
// THIS SPECIFICATION. The linkage is by convention only: the plugin's config
// header names this file. A reviewer verifying provenance must check it by hand.
package randomdisagg

// Config is the plugin's parameter surface. It mirrors RandomPDDeciderConfig in
// plugins/randomdisagg/pkg/epp/framework/plugins/scheduling/profilehandler/disagg/
// random_pd_decider.go; the JSON names there are the wire format.
type Config struct {
	// DisaggregateProbability is p, in [0,1]. 0 reproduces the simulator's
	// `never` rule, 1 its `always` rule. Values outside [0,1], and NaN, are
	// rejected at plugin construction rather than clamped -- a silently clamped
	// rate would misreport the arm's operating point.
	DisaggregateProbability float64

	// Seed selects which subset of requests is disaggregated at a given p.
	// Changing it at fixed p yields an independent draw of the same rate, which
	// is how to obtain replicate subsets without changing the arm.
	Seed int64

	// IdentityHeader, when non-empty and present on the request, is the string
	// U is keyed on. Empty (or absent on the request) falls back to
	// RequestID -- see R2.
	IdentityHeader string
}

// Policy is the arm. It holds no mutable state: there is no shadow table, no
// running estimate, and no RNG stream to advance. Two Policy values with equal
// Config decide identically for all requests.
type Policy struct {
	cfg Config
}

// ShouldDisaggregate is the arm's entire decision.
//
// IT DELIBERATELY DOES NOT IMPLEMENT THE CANDIDATE-RETURNING `Decide` THAT THE
// OTHER TWO ARMS IMPLEMENT. Those arms are pickers: they choose the (prefill,
// decode) pair themselves and so replace the target's whole scheduling graph.
// This arm chooses only WHETHER to disaggregate and leaves the choice of pods to
// the baseline's inherited load-shaped scorers (queue-scorer,
// kv-cache-utilization-scorer, prefix-cache-scorer, active-request-scorer).
//
// That is what makes it a matched control: the deployed difference between the
// baseline and this arm is ONE plugin -- `prefix-based-pd-decider` swapped for
// `random-pd-decider` -- and nothing else. Every filter, scorer, profile, engine
// flag, pod label, and objective is inherited unchanged. A difference in outcome
// is therefore attributable to the disaggregation decision alone.
func (p *Policy) ShouldDisaggregate(req Request) bool

// identity resolves the string U is keyed on: IdentityHeader's value when
// configured and present, else the Envoy-generated RequestID. See R2.
func (p *Policy) identity(req Request) string

// unitInterval maps (Seed, identity) deterministically into [0,1). Uses the top
// 53 bits of a 64-bit FNV-1a digest so the result is an exactly representable
// float64 and the comparison against p is not biased by rounding.
func (p *Policy) unitInterval(identity string) float64

// Request is the subset of the target's scheduling.InferenceRequest this arm
// reads. It is deliberately tiny: an identifier and headers. If a future revision
// of this file needs more, the arm has stopped being a control.
type Request struct {
	RequestID string
	Headers   map[string]string
}
